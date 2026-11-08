using OneSpeech.Evaluation;
using Xunit;
namespace OneSpeech.Evaluation.Tests;
public class EvaluationTests
{
    [Theory]
    [InlineData("Ten thousand dollars", "$10000", "Whisper Appendix C rule 9")]
    [InlineData("$68 million", "$68000000", "Our numeric scale policy")]
    [InlineData("$10,000", "$10000", "Whisper Appendix C rule 6")]
    [InlineData("twenty first", "21st", "Our ordinal spelling policy")]
    [InlineData("one hundred and five", "105", "Whisper Appendix C rule 9; our and policy")]
    [InlineData("three point five percent", "3.5%", "Our decimal/percent policy")]
    [InlineData("you're here, I'm not", "you are here i am not", "Whisper Appendix C rule 5")]
    [InlineData("she ’ll go", "she will go", "Whisper Appendix C rule 4")]
    [InlineData("can't won't don't", "can not will not do not", "Whisper Appendix C rule 5")]
    [InlineData("state-of-the-art", "state of the art", "Whisper Appendix C rule 8")]
    [InlineData("COLOUR and centre", "color and center", "Whisper Appendix C rule 10; our finite spelling table")]
    [InlineData("um hmm [noise] (laughs) résumé!", "resume", "Whisper Appendix C rules 1-3,8")]
    [InlineData("ah repeat repeat", "ah repeat repeat", "Our no semantic disfluency removal policy")]
    [InlineData("abc 123. next", "abc 123 next", "Whisper Appendix C rule 7")]
    [InlineData("", "", "Empty text")]
    public void English(string input,string expected,string rule) { Assert.NotEmpty(rule);Assert.Equal(expected,Normalizer.Normalize(input,"en")); }
    [Fact] public void Basic() => Assert.Equal("strasse σ 世界",Normalizer.Normalize("STRAẞE ς，世界！","basic"));
    [Fact] public void Counts()
    {
        Assert.Equal(new ErrorCounts(3,1,0,1),Metrics.Score("a b c","a x c d"));
        Assert.Equal(new ErrorCounts(3,0,2,0),Metrics.Score("a b c","a"));
        Assert.Equal(new ErrorCounts(0,0,0,1),Metrics.Score("","hello"));
        Assert.Null(Metrics.Score("","hello").Rate);
        Assert.Equal(new ErrorCounts(2,1,0,0),Metrics.Score("a 😀","a 😁",true));
    }
    [Fact] public void LinearCountsMatchAlignment()
    {
        string[] cases=["","a","b","a a","a b","b a","b b","a b a"];
        foreach(string a in cases)foreach(string b in cases)
            Assert.Equal(Metrics.Align(a.Split(' ',StringSplitOptions.RemoveEmptyEntries),b.Split(' ',StringSplitOptions.RemoveEmptyEntries)).Counts,Metrics.Score(a,b));
    }
    [Fact] public void PairedBootstrap()
    {
        ErrorCounts[] a=[new(10,1,0,0),new(10,2,0,0),new(10,0,0,0)];
        var first=Metrics.Bootstrap(a,a);var again=Metrics.Bootstrap(a,a);
        Assert.Equal(first,again);Assert.Equal(new Interval(0,0),first.Difference);Assert.Equal("within-noise",first.Verdict);
        Assert.Equal("within-fallback",Metrics.Verdict(null,.1,.105));
        Assert.Equal("different",Metrics.Verdict(new(.001,.01),.1,.102));
        Assert.Equal("not-comparable",Metrics.Verdict(null,null,.1));
        Assert.Throws<ArgumentException>(()=>Metrics.Bootstrap(a,[new(1,0,0,0)]));
    }
    [Fact] public void Timings()
    {
        var v=TimingMetrics.Score([new("hello",0,1),new("missing")],[new("hello",.05,1.1),new("missing")],"en");
        Assert.Equal(2,v.Matched);Assert.Equal(1,v.Timed);Assert.Equal(1,v.Missing);
        Assert.Equal(50,v.MeanStartMs!.Value,6);Assert.Equal(1,v.StartWithin50);Assert.Equal(1,v.EndWithin100);
    }
    [Fact] public void LoadersAndJson()
    {
        string dir=Path.Combine(Environment.GetEnvironmentVariable("TMPDIR")??throw new InvalidOperationException(),Guid.NewGuid().ToString("N"));Directory.CreateDirectory(dir);
        try
        {
            File.WriteAllText(Path.Combine(dir,"a.trans.txt"),"a HELLO WORLD\n");Assert.Equal("HELLO WORLD",Assert.Single(Hypotheses.LibriSpeech(dir)).Text);
            File.WriteAllText(Path.Combine(dir,"b.nlp"),"token|speaker|ts|endTs|punctuation|case|tags|wer_tags\nHello|0|0|1|!|UC|[]|[]\n");Assert.Equal("Hello",Assert.Single(Hypotheses.Earnings22(dir)).Text);
            string json=Path.Combine(dir,"output.jsonl");Hypotheses.Write(json,[new("a","Hi",[new("Hi",0,1)])]);Assert.Equal("Hi",Hypotheses.Read(json)["a"].Text);
            File.AppendAllText(json,File.ReadAllText(json));Assert.Throws<InvalidDataException>(()=>Hypotheses.Read(json));
            File.WriteAllText(json,"{\"id\":\"a\",\"text\":\"x\",\"words\":[{\"text\":\"x\",\"start\":2,\"end\":1}]}\n");Assert.Throws<InvalidDataException>(()=>Hypotheses.Read(json));
            foreach(string hostile in new[]{"null","[]","{\"id\":\"a\",\"text\":\"x\",\"words\":[null]}","{\"id\":\"a\"}"})
            { File.WriteAllText(json,hostile);Assert.ThrowsAny<Exception>(()=>Hypotheses.Read(json)); }
            string transcript=Path.Combine(dir,"a.json");File.WriteAllText(transcript,new OneSpeech.Transcript {Text="Hi"}.ToJson());Assert.Equal("Hi",Hypotheses.Read(transcript)["a"].Text);
        }
        finally {Directory.Delete(dir,true);}
    }
}
