using System.Collections.Immutable;

namespace OneSpeech.Decoding;

/// <summary>A tokenized contextual phrase. IDs must exclude blank and special tokens.</summary>
/// <param name="Tokens">One to 256 nonnegative token IDs, in emission order.</param>
/// <param name="Weight">Natural-log bonus per token, default 1; finite and in (0,20].</param>
public sealed record TokenPhrase(ImmutableArray<int> Tokens, double Weight = 1);

/// <summary>Immutable bounded token trie for inference-only contextual shallow fusion.</summary>
/// <remarks>Le et al. (2021) §3.1: add contextual scores before search pruning. This implementation
/// rewards completed phrases and lends partial-prefix credit that is repaid on mismatch or finalization.
/// Overlapping completions at one token receive the largest reward, not their sum. Duplicate phrases
/// use the largest weight. Matching is on token IDs, not text or word boundaries; callers tokenize
/// boundary-marked phrases and must not supply blank or control IDs. No model parameters are changed.</remarks>
public sealed class PhraseTrie
{
    private sealed class Node
    {
        internal readonly Dictionary<int, int> Children = [];
        internal int Failure;
        internal double Potential;
        internal double Reward;
    }
    private readonly Node[] nodes;

    /// <summary>Builds up to 4096 phrases and 65536 total tokens. Empty input disables biasing.</summary>
    /// <exception cref="ArgumentException">A phrase, token, weight or resource bound is invalid.</exception>
    public PhraseTrie(IEnumerable<TokenPhrase> phrases)
    {
        ArgumentNullException.ThrowIfNull(phrases);
        List<Node> building = [new()];
        int count = 0, tokens = 0;
        foreach (var phrase in phrases)
        {
            if (phrase is null || phrase.Tokens.IsDefaultOrEmpty || phrase.Tokens.Length > 256 ||
                !double.IsFinite(phrase.Weight) || phrase.Weight is <= 0 or > 20 || ++count > 4096 ||
                (tokens += phrase.Tokens.Length) > 65536)
                throw new ArgumentException("Invalid contextual phrase or phrase-list bound.", nameof(phrases));
            int state = 0;
            for (int i = 0; i < phrase.Tokens.Length; i++)
            {
                int token = phrase.Tokens[i];
                if (token is < 0 or >= 1000000) throw new ArgumentException("Phrase token is outside 0..999999.", nameof(phrases));
                if (!building[state].Children.TryGetValue(token, out int next))
                {
                    next = building.Count; building[state].Children.Add(token, next); building.Add(new());
                }
                state = next;
                if (i + 1 < phrase.Tokens.Length) building[state].Potential = Math.Max(building[state].Potential, (i + 1) * phrase.Weight);
            }
            building[state].Reward = Math.Max(building[state].Reward, phrase.Tokens.Length * phrase.Weight);
        }
        nodes = building.ToArray();
        Queue<int> queue = new();
        foreach (int child in nodes[0].Children.Values) queue.Enqueue(child);
        while (queue.TryDequeue(out int state))
        {
            var node = nodes[state];
            node.Reward = Math.Max(node.Reward, nodes[node.Failure].Reward);
            node.Potential = Math.Max(node.Potential, nodes[node.Failure].Potential);
            foreach (var (token, child) in node.Children)
            {
                int failure = node.Failure;
                while (failure != 0 && !nodes[failure].Children.ContainsKey(token)) failure = nodes[failure].Failure;
                nodes[child].Failure = nodes[failure].Children.TryGetValue(token, out int target) ? target : 0;
                queue.Enqueue(child);
            }
        }
    }

    /// <summary>Advances a nonblank token from state (initially zero), returning a natural-log score delta.
    /// Blank/repeated CTC frames must not call this method. States belong only to this trie.</summary>
    /// <exception cref="ArgumentOutOfRangeException">State or token is outside its bounds.</exception>
    public double Advance(int state, int token, out int nextState)
    {
        ValidateState(state);
        if (token is < 0 or >= 1000000) throw new ArgumentOutOfRangeException(nameof(token));
        int next = state;
        while (next != 0 && !nodes[next].Children.ContainsKey(token)) next = nodes[next].Failure;
        nextState = nodes[next].Children.TryGetValue(token, out int child) ? child : 0;
        return nodes[nextState].Reward + nodes[nextState].Potential - nodes[state].Potential;
    }

    /// <summary>Returns the finalization delta, repaying any incomplete-prefix credit. Does not mutate state.</summary>
    /// <exception cref="ArgumentOutOfRangeException">State is outside this trie.</exception>
    public double Finish(int state) { ValidateState(state); return -nodes[state].Potential; }

    /// <summary>Scores a complete nonblank sequence; partial phrases receive zero net credit. Empty input scores zero.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A token is outside 0..999999.</exception>
    public double Score(ReadOnlySpan<int> tokens)
    {
        int state = 0; double score = 0;
        foreach (int token in tokens) { score += Advance(state, token, out int next); state = next; }
        return score + Finish(state);
    }

    internal void ValidateVocabulary(int vocabularySize, int blankId)
    {
        foreach (var node in nodes)
            foreach (int token in node.Children.Keys)
                if (token >= vocabularySize || token == blankId) throw new ArgumentException("Contextual phrases contain blank or an out-of-vocabulary token.");
    }

    private void ValidateState(int state)
    {
        if ((uint)state >= nodes.Length) throw new ArgumentOutOfRangeException(nameof(state));
    }
}
