namespace OneSpeech.Diarization;

// Median-of-three partition selection. Tied scores are partitioned, rather than all favoring early audio.
internal static class CacheTopScores
{
    internal static int[] Select(float[] scores, int count, bool previousElement = true)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(count, scores.Length);
        int[] indices = Enumerable.Range(0, scores.Length).ToArray();
        int left = 0, right = indices.Length, nth = previousElement ? count - 1 : count;
        if (nth == right) return indices;
        int depth = 2 * (int)Math.Log2(indices.Length);
        while (right - left > 3)
        {
            if (depth-- == 0)
            {
                Array.Sort(indices, left, right - left, Comparer<int>.Create((a, b) => scores[b].CompareTo(scores[a])));
                return indices.AsSpan(0, count).ToArray();
            }
            int a = left + 1, b = left + (right - left) / 2, c = right - 1;
            int median;
            if (Greater(indices[a], indices[b])) median = Greater(indices[b], indices[c]) ? b : Greater(indices[a], indices[c]) ? c : a;
            else median = Greater(indices[a], indices[c]) ? a : Greater(indices[b], indices[c]) ? c : b;
            Swap(left, median);
            float pivot = scores[indices[left]];
            int first = left + 1, last = right;
            while (true)
            {
                while (first < last && scores[indices[first]] > pivot) first++;
                do { last--; } while (last >= first && pivot > scores[indices[last]]);
                if (first >= last) break;
                Swap(first++, last);
            }
            if (first <= nth) left = first; else right = first;
        }
        for (int i = left + 1; i < right; i++)
        {
            int value = indices[i], j = i;
            while (j > left && Greater(value, indices[j - 1])) { indices[j] = indices[j - 1]; j--; }
            indices[j] = value;
        }
        return indices.AsSpan(0, count).ToArray();
        bool Greater(int a, int b) => scores[a] > scores[b];
        void Swap(int a, int b) => (indices[a], indices[b]) = (indices[b], indices[a]);
    }
}
