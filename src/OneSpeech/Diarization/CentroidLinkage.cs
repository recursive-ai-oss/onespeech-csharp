namespace OneSpeech.Diarization;

/// <summary>Agglomerative clustering with centroid linkage on Euclidean distance, stopped at a distance threshold.
/// Merges follow the Lance-Williams update for squared distances: d2(k, i+j) = (ni d2(k,i) + nj d2(k,j)) / (ni + nj)
/// - ni nj d2(i,j) / (ni + nj)^2. Memory is O(n^2) floats; callers bound n.</summary>
internal static class CentroidLinkage
{
    internal static int[] Cluster(float[][] points, double threshold)
    {
        int n = points.Length;
        if (n == 0) return [];
        var d2 = new float[n][];
        for (int i = 0; i < n; i++)
        {
            d2[i] = new float[n];
            for (int j = 0; j < i; j++)
            {
                double s = 0;
                for (int k = 0; k < points[i].Length; k++) { double d = points[i][k] - points[j][k]; s += d * d; }
                d2[i][j] = d2[j][i] = (float)s;
            }
        }
        var size = new int[n]; var alive = new bool[n]; var parent = new int[n];
        for (int i = 0; i < n; i++) { size[i] = 1; alive[i] = true; parent[i] = i; }
        var nearest = new int[n]; var nearestD = new float[n];
        void Refresh(int i)
        {
            nearest[i] = -1; nearestD[i] = float.PositiveInfinity;
            for (int j = 0; j < n; j++) if (j != i && alive[j] && d2[i][j] < nearestD[i]) { nearestD[i] = d2[i][j]; nearest[i] = j; }
        }
        for (int i = 0; i < n; i++) Refresh(i);
        double limit = threshold * threshold;
        while (true)
        {
            int a = -1; float best = float.PositiveInfinity;
            for (int i = 0; i < n; i++) if (alive[i] && nearestD[i] < best) { best = nearestD[i]; a = i; }
            if (a < 0 || best > limit) break;
            int b = nearest[a];
            double na = size[a], nb = size[b], total = na + nb;
            for (int k = 0; k < n; k++)
            {
                if (!alive[k] || k == a || k == b) continue;
                float v = (float)((na * d2[k][a] + nb * d2[k][b]) / total - na * nb * d2[a][b] / (total * total));
                d2[k][a] = d2[a][k] = Math.Max(v, 0);
            }
            alive[b] = false; size[a] += size[b]; parent[b] = a;
            for (int i = 0; i < n; i++)
                if (alive[i] && (i == a || nearest[i] == a || nearest[i] == b)) Refresh(i);
                else if (alive[i] && d2[i][a] < nearestD[i]) { nearestD[i] = d2[i][a]; nearest[i] = a; }
        }
        var labels = new int[n]; var ids = new Dictionary<int, int>();
        for (int i = 0; i < n; i++)
        {
            int root = i; while (parent[root] != root) root = parent[root];
            if (!ids.TryGetValue(root, out int id)) ids[root] = id = ids.Count;
            labels[i] = id;
        }
        return labels;
    }
}
