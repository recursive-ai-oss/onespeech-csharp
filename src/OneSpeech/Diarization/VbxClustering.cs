namespace OneSpeech.Diarization;

/// <summary>Variational Bayes HMM clustering of PLDA-space x-vectors (Landini et al. 2022, arXiv 2012.14952 §§2.1-2.5).
/// States are speakers, transitions stay with probability <c>loopP</c>, speaker priors pi are re-estimated and drive
/// redundant speakers to zero. Inputs are rho = sqrt(phi) * x per frame; no parameter is learned from data here.</summary>
internal static class VbxClustering
{
    internal sealed record Result(double[][] Responsibilities, double[] Priors, double Elbo, int Iterations);

    // features: T frames of D-dimensional x-vectors scaled by sqrt(phi) (paper eq. 18); phi: between-speaker variances;
    // labels: initial hard labels in [0, speakers) from agglomerative clustering; fa, fb: the F_A and F_B scaling factors
    // (0.07 and 0.8 in the published configuration); loopP: P_loop of eq. 1; smoothing: initial responsibilities are
    // softmax(smoothing * one-hot(label)).
    internal static Result Run(double[][] features, double[] phi, int[] labels, int speakers, double fa, double fb, double loopP, double smoothing, int maxIterations, double epsilon)
    {
        int T = features.Length, D = phi.Length;
        if (T < 2 || speakers < 1) throw new ArgumentException("VBx needs at least two frames and one speaker.");
        var gamma = new double[T][];
        double hot = Math.Exp(smoothing);
        for (int t = 0; t < T; t++)
        {
            gamma[t] = new double[speakers];
            double z = hot + speakers - 1;
            for (int s = 0; s < speakers; s++) gamma[t][s] = (s == labels[t] ? hot : 1) / z;
        }
        var pi = new double[speakers];
        Array.Fill(pi, 1.0 / speakers);
        var logp = new double[T][];
        for (int t = 0; t < T; t++) logp[t] = new double[speakers];
        var alpha = new double[T][]; var beta = new double[T][]; var scale = new double[T];
        for (int t = 0; t < T; t++) { alpha[t] = new double[speakers]; beta[t] = new double[speakers]; }
        double previous = double.NegativeInfinity, elbo = 0;
        int iteration = 0;
        var ybar = new double[speakers][]; var ell = new double[speakers][];
        for (int s = 0; s < speakers; s++) { ybar[s] = new double[D]; ell[s] = new double[D]; }
        var shift = new double[T];
        for (; iteration < maxIterations; iteration++)
        {
            // q(y) update, eqs. 16-17 with diagonal Phi.
            double klTerm = 0;
            for (int s = 0; s < speakers; s++)
            {
                double ns = 0;
                for (int t = 0; t < T; t++) ns += gamma[t][s];
                var sum = new double[D];
                for (int t = 0; t < T; t++) { double g = gamma[t][s]; if (g < 1e-12) continue; for (int d = 0; d < D; d++) sum[d] += g * features[t][d]; }
                for (int d = 0; d < D; d++)
                {
                    double invL = 1.0 / (1.0 + fa / fb * ns * phi[d]);
                    ell[s][d] = invL;
                    ybar[s][d] = fa / fb * invL * sum[d];
                    klTerm += 1 + Math.Log(invL) - invL - ybar[s][d] * ybar[s][d];
                }
            }
            // Expected log-likelihood per frame and speaker, eq. 23 (terms common to all speakers dropped).
            for (int t = 0; t < T; t++)
            {
                double max = double.NegativeInfinity;
                for (int s = 0; s < speakers; s++)
                {
                    double dot = 0, pen = 0;
                    for (int d = 0; d < D; d++) { dot += ybar[s][d] * features[t][d]; pen += phi[d] * (ell[s][d] + ybar[s][d] * ybar[s][d]); }
                    logp[t][s] = fa * (dot - 0.5 * pen);
                    if (logp[t][s] > max) max = logp[t][s];
                }
                shift[t] = max;
                for (int s = 0; s < speakers; s++) logp[t][s] = Math.Exp(logp[t][s] - max);
            }
            // Scaled forward pass (eq. 20) with transitions p(s|s') = (1-loopP) pi_s + loopP [s = s'].
            double logLikelihood = 0;
            for (int t = 0; t < T; t++)
            {
                double total = 0, previousSum = 1;
                if (t > 0) { previousSum = 0; for (int s = 0; s < speakers; s++) previousSum += alpha[t - 1][s]; }
                for (int s = 0; s < speakers; s++)
                {
                    double enter = t == 0 ? pi[s] : (1 - loopP) * pi[s] * previousSum + loopP * alpha[t - 1][s];
                    alpha[t][s] = logp[t][s] * enter;
                    total += alpha[t][s];
                }
                scale[t] = total;
                for (int s = 0; s < speakers; s++) alpha[t][s] /= total;
                logLikelihood += Math.Log(total) + shift[t];
            }
            // Backward pass (eq. 21) and responsibilities (eq. 19).
            for (int s = 0; s < speakers; s++) beta[T - 1][s] = 1;
            for (int t = T - 2; t >= 0; t--)
            {
                double mix = 0;
                for (int s = 0; s < speakers; s++) mix += pi[s] * logp[t + 1][s] * beta[t + 1][s];
                for (int s = 0; s < speakers; s++) beta[t][s] = ((1 - loopP) * mix + loopP * logp[t + 1][s] * beta[t + 1][s]) / scale[t + 1];
            }
            for (int t = 0; t < T; t++)
            {
                double z = 0;
                for (int s = 0; s < speakers; s++) { gamma[t][s] = alpha[t][s] * beta[t][s]; z += gamma[t][s]; }
                for (int s = 0; s < speakers; s++) gamma[t][s] /= z;
            }
            // Speaker priors, eq. 24.
            var next = new double[speakers];
            double norm = 0;
            for (int s = 0; s < speakers; s++)
            {
                double extra = 0;
                for (int t = 1; t < T; t++) extra += logp[t][s] * beta[t][s] / scale[t];
                next[s] = gamma[0][s] + (1 - loopP) * pi[s] * extra;
                norm += next[s];
            }
            for (int s = 0; s < speakers; s++) pi[s] = next[s] / norm;
            // ELBO, eq. 25.
            elbo = logLikelihood + fb / 2 * klTerm;
            if (iteration > 0 && elbo - previous < epsilon) { iteration++; break; }
            previous = elbo;
        }
        return new Result(gamma, pi, elbo, iteration);
    }
}
