namespace OneSpeech.Decoding;

/// <summary>Monotone minimum-cost warping over a row-major cost lattice.</summary>
public static class DynamicTimeWarping
{
    /// <summary>Finds the minimum-cost path from (0,0) to (rows-1,columns-1), allowing diagonal, row and column advances. Exact ties prefer diagonal, then row advance.</summary>
    /// <remarks>Rows and columns must be positive, their product at most 16 million, and costs finite. Returned pairs are in forward order. Cancellation is checked per row.</remarks>
    /// <exception cref="ArgumentException">Shape or costs are invalid.</exception>
    /// <exception cref="OperationCanceledException">Cancellation is requested.</exception>
    public static (int Row, int Column)[] Align(ReadOnlySpan<float> costs, int rows, int columns, CancellationToken cancellationToken = default)
    {
        if (rows <= 0 || columns <= 0 || (long)rows * columns > 16000000 || (long)rows * columns != costs.Length) throw new ArgumentException("Invalid DTW shape.");
        byte[] trace = new byte[costs.Length];
        double[] previous = new double[columns + 1], current = new double[columns + 1];
        Array.Fill(previous, double.PositiveInfinity); previous[0] = 0;
        for (int row = 0; row < rows; row++)
        {
            cancellationToken.ThrowIfCancellationRequested(); current[0] = double.PositiveInfinity;
            for (int col = 0; col < columns; col++)
            {
                float cost = costs[row * columns + col];
                if (!float.IsFinite(cost)) throw new ArgumentException("DTW costs must be finite.", nameof(costs));
                double best = previous[col]; byte direction = 0;
                if (previous[col + 1] < best) { best = previous[col + 1]; direction = 1; }
                if (current[col] < best) { best = current[col]; direction = 2; }
                current[col + 1] = best + cost; trace[row * columns + col] = direction;
            }
            (previous, current) = (current, previous);
        }
        List<(int, int)> path = [];
        int r = rows - 1, c = columns - 1;
        while (r >= 0 && c >= 0)
        {
            path.Add((r, c));
            byte direction = trace[r * columns + c];
            if (direction != 2) r--;
            if (direction != 1) c--;
        }
        path.Reverse(); return path.ToArray();
    }
}
