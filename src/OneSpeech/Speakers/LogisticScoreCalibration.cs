namespace OneSpeech.Speakers;

/// <summary>Deterministic affine logistic score calibration fitted only on labeled development scores.
/// This calibrates scores, not model weights. No inputs are retained or persisted.</summary>
public sealed class LogisticScoreCalibration
{
    private readonly double center;
    private readonly double scale;
    private readonly double slope;
    private readonly double intercept;

    private LogisticScoreCalibration(double center, double scale, double slope, double intercept)
    {
        this.center = center;
        this.scale = scale;
        this.slope = slope;
        this.intercept = intercept;
    }

    /// <summary>Fits empirical-prior, equally weighted binary cross-entropy with L2 penalty on both coefficients.
    /// Labels are true for same-speaker trials. Scores must be finite with absolute value at most 1e6;
    /// both classes must occur and lengths must match. Scores are standardized by population deviation
    /// (floor 1e-6). regularization defaults to 1e-4 and must be finite and positive;
    /// iterations defaults to 100 and must be in [1,10000]. Invalid input throws ArgumentException.
    /// Cancellation throws OperationCanceledException. No random initialization is used.</summary>
    public static LogisticScoreCalibration Fit(ReadOnlySpan<double> scores, ReadOnlySpan<bool> sameSpeaker,
        CancellationToken cancellationToken, double regularization = 1e-4, int iterations = 100)
    {
        if (scores.IsEmpty || scores.Length != sameSpeaker.Length) throw new ArgumentException("Nonempty scores and matching labels are required.", nameof(scores));
        if (!double.IsFinite(regularization) || regularization <= 0) throw new ArgumentOutOfRangeException(nameof(regularization));
        if (iterations is < 1 or > 10000) throw new ArgumentOutOfRangeException(nameof(iterations));
        int positives = 0;
        double mean = 0, variance = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!double.IsFinite(scores[i]) || Math.Abs(scores[i]) > 1e6) throw new ArgumentException("Scores must be finite and bounded by 1e6.", nameof(scores));
            mean += scores[i] / scores.Length;
            if (sameSpeaker[i]) positives++;
        }
        if (positives == 0 || positives == scores.Length) throw new ArgumentException("Both classes are required.", nameof(sameSpeaker));
        foreach (double score in scores) variance += (score - mean) * (score - mean) / scores.Length;
        double scale = Math.Max(1e-6, Math.Sqrt(variance));
        double a = 0, b = 0;
        // Fixed-step descent: the standardized design has mean square one, so the logistic
        // Hessian's largest eigenvalue is bounded by 0.5 plus the regularizer.
        double step = 1 / (0.5 + regularization);
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            double ga = regularization * a, gb = regularization * b;
            for (int i = 0; i < scores.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                double x = (scores[i] - mean) / scale;
                double error = Sigmoid(a * x + b) - (sameSpeaker[i] ? 1 : 0);
                ga += error * x / scores.Length;
                gb += error / scores.Length;
            }
            a -= step * ga;
            b -= step * gb;
        }
        return new LogisticScoreCalibration(mean, scale, a, b);
    }

    /// <summary>Returns a probability under the development set's class prior, not a population identity guarantee.
    /// Throws ArgumentOutOfRangeException for nonfinite scores or absolute values above 1e6.</summary>
    public double Apply(double score)
    {
        if (!double.IsFinite(score) || Math.Abs(score) > 1e6) throw new ArgumentOutOfRangeException(nameof(score));
        return Sigmoid(slope * ((score - center) / scale) + intercept);
    }

    private static double Sigmoid(double value) => value >= 0 ? 1 / (1 + Math.Exp(-value)) : Math.Exp(value) / (1 + Math.Exp(value));
}
