using System.Text.Json;
using OneSpeech.Evaluation;
using Xunit;

namespace OneSpeech.Evaluation.Tests;

public sealed class TimingFloorTests
{
    [Fact]
    public void PooledSummaryDoesNotRealignTheCorpusInOneQuadraticTrellis()
    {
        string root = Path.Combine(Environment.GetEnvironmentVariable("TMPDIR") ?? Path.GetTempPath(), "timing-floor-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            var records = Enumerable.Range(0, 32).Select(i => new Utterance(i.ToString(System.Globalization.CultureInfo.InvariantCulture), "synthetic", Enumerable.Range(0, 128).Select(j => new EvalWord("token" + j.ToString(System.Globalization.CultureInfo.InvariantCulture), j * .01, j * .01 + .005)).ToArray())).ToArray();
            foreach (string name in new[] { "truth", "ctc", "dtw", "oracle" }) Hypotheses.Write(Path.Combine(root, name + ".jsonl"), records);
            Summary.Write(root);
            using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(root, "summary.json")));
            Assert.Equal(4096, document.RootElement.GetProperty("commonTimedWords").GetInt32());
            Assert.Equal(0, document.RootElement.GetProperty("ctc").GetProperty("common").GetProperty("meanStartMs").GetDouble());
        }
        finally { Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public void SummaryPoolsWordsAndSeparatesMissingUnmatchedAndCommonCoverage()
    {
        string root = Path.Combine(Environment.GetEnvironmentVariable("TMPDIR") ?? Path.GetTempPath(), "timing-floor-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            Hypotheses.Write(Path.Combine(root, "truth.jsonl"), [new("a", "one two three", [new("one", 0, 1), new("two", 1, 2), new("three", 2, 3)]), new("b", "one", [new("one", 0, 1)])]);
            Hypotheses.Write(Path.Combine(root, "ctc.jsonl"), [new("a", "one two three", [new("one", .02, 1.02), new("two"), new("three", 2.2, 3.2)]), new("b", "one", [new("one", .04, 1.04)])]);
            Hypotheses.Write(Path.Combine(root, "dtw.jsonl"), [new("a", "one two", [new("one", .06, 1.06), new("two", 1, 2)]), new("b", "one", [new("one", .1, 1.1)])]);
            Hypotheses.Write(Path.Combine(root, "oracle.jsonl"), [new("a", "one two three", [new("one", .03, 1.03), new("two", 1, 2), new("three", 2, 3)]), new("b", "one", [new("one", .05, 1.05)])]);
            Summary.Write(root);
            using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(root, "summary.json"))); var result = document.RootElement;
            Assert.Equal(4, result.GetProperty("truthWords").GetInt32());
            Assert.Equal(2, result.GetProperty("commonTimedWords").GetInt32());
            Assert.Equal(1, result.GetProperty("ctc").GetProperty("hypothesisUntimed").GetInt32());
            Assert.Equal(1, result.GetProperty("dtw").GetProperty("unmatchedTruthWords").GetInt32());
            Assert.Equal(3, result.GetProperty("ctc").GetProperty("allMatched").GetProperty("timed").GetInt32());
            Assert.Equal(30, result.GetProperty("ctc").GetProperty("common").GetProperty("meanStartMs").GetDouble(), 8);
            Assert.Equal(30, result.GetProperty("ctc").GetProperty("common").GetProperty("medianStartMs").GetDouble(), 8);
            Assert.Equal(80, result.GetProperty("dtw").GetProperty("common").GetProperty("meanStartMs").GetDouble(), 8);
        }
        finally { Directory.Delete(root, recursive: true); }
    }
}
