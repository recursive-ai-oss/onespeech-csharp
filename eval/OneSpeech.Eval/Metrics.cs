namespace OneSpeech.Evaluation;

public sealed record ErrorCounts(long Reference, long Substitutions, long Deletions, long Insertions)
{
    public long Errors => Substitutions + Deletions + Insertions;
    public double? Rate => Reference == 0 ? null : (double)Errors / Reference;
    public static ErrorCounts operator +(ErrorCounts a, ErrorCounts b) => new(a.Reference+b.Reference,a.Substitutions+b.Substitutions,a.Deletions+b.Deletions,a.Insertions+b.Insertions);
}
public sealed record Match(int Reference, int Hypothesis, bool Equal);
public sealed record Alignment(ErrorCounts Counts, IReadOnlyList<Match> Matches);
public sealed record Interval(double Lower, double Upper);
public sealed record BootstrapResult(Interval? First, Interval? Second, Interval? Difference, int Resamples, ulong Seed, string Verdict);

public static class Metrics
{
    // Deterministic tie order: diagonal, deletion, insertion. Full trace only for word matching.
    public static Alignment Align(IReadOnlyList<string> reference, IReadOnlyList<string> hypothesis)
    {
        if ((long)(reference.Count+1)*(hypothesis.Count+1)>16_000_000) throw new ArgumentException("Alignment exceeds 16 million cells; segment the utterance.");
        var trace=new byte[reference.Count+1,hypothesis.Count+1];
        var prev=new int[hypothesis.Count+1]; var next=new int[prev.Length];
        for(int j=1;j<prev.Length;j++) { prev[j]=j; trace[0,j]=2; }
        for(int i=1;i<=reference.Count;i++)
        {
            next[0]=i; trace[i,0]=1;
            for(int j=1;j<prev.Length;j++)
            {
                int best=prev[j-1]+(reference[i-1]==hypothesis[j-1]?0:1); byte action=0;
                if(prev[j]+1<best) {best=prev[j]+1;action=1;}
                if(next[j-1]+1<best) {best=next[j-1]+1;action=2;}
                next[j]=best;trace[i,j]=action;
            }
            (prev,next)=(next,prev);
        }
        long s=0,d=0,ins=0; int x=reference.Count,y=hypothesis.Count; var matches=new List<Match>();
        while(x>0 || y>0)
        {
            switch(trace[x,y])
            {
                case 0: x--;y--;bool equal=reference[x]==hypothesis[y];if(!equal)s++;matches.Add(new(x,y,equal));break;
                case 1:x--;d++;break;
                default:y--;ins++;break;
            }
        }
        matches.Reverse();return new(new(reference.Count,s,d,ins),matches);
    }
    public static ErrorCounts Score(string reference,string hypothesis,bool characters=false)
    {
        string[] Tokens(string s) => characters ? s.EnumerateRunes().Where(r=>!System.Text.Rune.IsWhiteSpace(r)).Select(r=>r.ToString()).ToArray() : s.Split(' ',StringSplitOptions.RemoveEmptyEntries);
        var a=Tokens(reference);var b=Tokens(hypothesis);
        var previous=new (long S,long D,long I)[b.Length+1];var current=new (long S,long D,long I)[b.Length+1];
        for(int j=0;j<=b.Length;j++)previous[j]=(0,0,j);
        for(int i=1;i<=a.Length;i++)
        {
            current[0]=(0,i,0);
            for(int j=1;j<=b.Length;j++)
            {
                var p=previous[j-1];var best=(S:p.S+(a[i-1]==b[j-1]?0:1),D:p.D,I:p.I);
                p=previous[j];var deletion=(S:p.S,D:p.D+1,I:p.I);
                p=current[j-1];var insertion=(S:p.S,D:p.D,I:p.I+1);
                if(deletion.S+deletion.D+deletion.I<best.S+best.D+best.I)best=deletion;
                if(insertion.S+insertion.D+insertion.I<best.S+best.D+best.I)best=insertion;
                current[j]=best;
            }
            (previous,current)=(current,previous);
        }
        var final=previous[b.Length];return new(a.Length,final.S,final.D,final.I);
    }
    public static BootstrapResult Bootstrap(IReadOnlyList<ErrorCounts> first,IReadOnlyList<ErrorCounts>? second=null,int resamples=1000,ulong seed=1729)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(resamples,1000);
        if(second is not null && second.Count!=first.Count)throw new ArgumentException("Paired utterance counts differ.");
        ErrorCounts Sum(IReadOnlyList<ErrorCounts> c)=>c.Aggregate(new ErrorCounts(0,0,0,0),(a,b)=>a+b);
        var a=Sum(first);var b=second is null?null:Sum(second);
        if(second is not null && first.Where((v,i)=>v.Reference!=second[i].Reference).Any())throw new ArgumentException("Paired reference counts differ.");
        if(first.Count<2 || first.Any(v=>v.Reference==0))return new(null,null,null,resamples,seed,Verdict(null,a.Rate,b?.Rate));
        ulong state=seed; var aa=new double[resamples];var bb=new double[resamples];var dd=new double[resamples];
        for(int r=0;r<resamples;r++)
        {
            long n=0,e1=0,e2=0;
            for(int i=0;i<first.Count;i++)
            {
                // SplitMix64: fixed integer algorithm makes the seed portable across runtimes.
                state+=0x9E3779B97F4A7C15UL;ulong z=state;z=(z^(z>>30))*0xBF58476D1CE4E5B9UL;z=(z^(z>>27))*0x94D049BB133111EBUL;z^=z>>31;
                int k=(int)(z%(uint)first.Count);n+=first[k].Reference;e1+=first[k].Errors;if(second is not null)e2+=second[k].Errors;
            }
            aa[r]=(double)e1/n;bb[r]=(double)e2/n;dd[r]=aa[r]-bb[r];
        }
        Interval CI(double[] v){Array.Sort(v);return new(v[(int)Math.Floor(.025*(v.Length-1))],v[(int)Math.Ceiling(.975*(v.Length-1))]);}
        var delta=second is null?null:CI(dd);
        return new(CI(aa),second is null?null:CI(bb),delta,resamples,seed,Verdict(delta,a.Rate,b?.Rate));
    }
    public static string Verdict(Interval? difference,double? first,double? second) => first is null || second is null ? "not-comparable" : difference is not null ? difference.Lower<=0 && difference.Upper>=0 ? "within-noise" : "different" : Math.Abs(first.Value-second.Value)<=.005+1e-12 ? "within-fallback" : "different";
}

public sealed record BoundarySummary(int Matched,int Timed,int Missing,double? MeanStartMs,double? MedianStartMs,double? MeanEndMs,double? MedianEndMs,double? StartWithin50,double? StartWithin100,double? EndWithin50,double? EndWithin100);
public static class TimingMetrics
{
    public static BoundarySummary Score(IReadOnlyList<EvalWord> reference,IReadOnlyList<EvalWord> hypothesis,string normalizer)
    {
        var alignment=Metrics.Align(reference.Select(w=>Normalizer.Normalize(w.Text,normalizer)).ToArray(),hypothesis.Select(w=>Normalizer.Normalize(w.Text,normalizer)).ToArray());
        var start=new List<double>();var end=new List<double>();int matched=0;
        foreach(var m in alignment.Matches.Where(m=>m.Equal && Normalizer.Normalize(reference[m.Reference].Text,normalizer).Length>0))
        {
            matched++;var a=reference[m.Reference];var b=hypothesis[m.Hypothesis];
            if(a.Start is null || b.Start is null || a.End is null || b.End is null)continue;
            start.Add(Math.Abs(a.Start.Value-b.Start.Value)*1000);end.Add(Math.Abs(a.End.Value-b.End.Value)*1000);
        }
        double? Mean(List<double> x)=>x.Count==0?null:x.Average();
        double? Median(List<double> x){if(x.Count==0)return null;var v=x.Order().ToArray();return (v[(v.Length-1)/2]+v[v.Length/2])/2;}
        double? Within(List<double> x,int ms)=>x.Count==0?null:(double)x.Count(v=>v<=ms+1e-9)/x.Count;
        return new(matched,start.Count,matched-start.Count,Mean(start),Median(start),Mean(end),Median(end),Within(start,50),Within(start,100),Within(end,50),Within(end,100));
    }
}
