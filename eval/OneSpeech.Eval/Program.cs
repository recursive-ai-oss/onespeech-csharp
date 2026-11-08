using System.Text.Json;
using OneSpeech.Evaluation;

try
{
    if(args.Length==0)throw new ArgumentException("Expected score or references.");
    var options=new Dictionary<string,string>(StringComparer.Ordinal);
    for(int i=1;i<args.Length;i+=2) {if(i+1>=args.Length || !args[i].StartsWith("--",StringComparison.Ordinal) || !options.TryAdd(args[i],args[i+1]))throw new ArgumentException("Expected unique --option value pairs.");}
    string Required(string name)=>options.GetValueOrDefault(name)??throw new ArgumentException("Missing "+name);
    if(args[0]=="references")
    {
        var records=Required("--dataset") switch {"librispeech"=>Hypotheses.LibriSpeech(Required("--input")),"earnings22"=>Hypotheses.Earnings22(Required("--input")),_=>throw new ArgumentException("Unknown dataset.")};
        Hypotheses.Write(Required("--out"),records);return 0;
    }
    if(args[0]=="der")
    {
        IReadOnlyList<DiarizationSegment> Load(string path)
        {
            if(Path.GetExtension(path).Equals(".json",StringComparison.OrdinalIgnoreCase))
                return Rttm.FromTranscript(OneSpeech.Transcript.FromJson(File.ReadAllBytes(path)),options.GetValueOrDefault("--recording","recording"));
            using var input=File.OpenRead(path);
            return Rttm.Read(input);
        }
        var score=DiarizationMetrics.Score(Load(Required("--ref")),Load(Required("--hyp")));
        string derOutput=options.GetValueOrDefault("--out","der.json");
        using(var file=File.Create(derOutput))
        using(var writer=new Utf8JsonWriter(file,new JsonWriterOptions{Indented=true}))
        {
            writer.WriteStartObject();writer.WriteNumber("referenceSeconds",score.ReferenceSeconds);
            writer.WriteNumber("missSeconds",score.MissSeconds);writer.WriteNumber("falseAlarmSeconds",score.FalseAlarmSeconds);
            writer.WriteNumber("confusionSeconds",score.ConfusionSeconds);writer.WriteNumber("referenceSpeakers",score.ReferenceSpeakers);
            if(score.Der is {} der)writer.WriteNumber("der",der);else writer.WriteNull("der");
            if(score.Jer is {} jer)writer.WriteNumber("jer",jer);else writer.WriteNull("jer");
            writer.WriteEndObject();
        }
        Console.WriteLine("Result: "+derOutput);return 0;
    }
    if(args[0]!="score")throw new ArgumentException("Unknown command.");
    var refs=Hypotheses.Read(Required("--ref"));var hyp=Hypotheses.Read(Required("--hyp"));
    var second=options.TryGetValue("--hyp2",out var h2)?Hypotheses.Read(h2):null;
    void SameIds(Dictionary<string,Utterance> h) {if(refs.Count!=h.Count || refs.Keys.Any(id=>!h.ContainsKey(id)))throw new InvalidDataException("Utterance IDs must match exactly; missing or extra hypotheses are not silently discarded.");}
    SameIds(hyp);if(second is not null)SameIds(second);
    string normalizer=options.GetValueOrDefault("--normalizer","en");
    string N(string s)=>Normalizer.Normalize(s,normalizer);
    var rows=refs.OrderBy(p=>p.Key,StringComparer.Ordinal).Select(p=>new UtteranceScore(p.Key,Metrics.Score(N(p.Value.Text),N(hyp[p.Key].Text)),Metrics.Score(N(p.Value.Text),N(hyp[p.Key].Text),true),second is null?null:Metrics.Score(N(p.Value.Text),N(second[p.Key].Text)),p.Value.Words is null || hyp[p.Key].Words is null || !p.Value.Words.Any(w=>w.Start is not null) || !hyp[p.Key].Words!.Any(w=>w.Start is not null)?null:TimingMetrics.Score(p.Value.Words,hyp[p.Key].Words!,normalizer))).ToArray();
    ErrorCounts Sum(IEnumerable<ErrorCounts> v)=>v.Aggregate(new ErrorCounts(0,0,0,0),(a,b)=>a+b);
    var bootstrap=Metrics.Bootstrap(rows.Select(r=>r.Wer).ToArray(),second is null?null:rows.Select(r=>r.SecondWer!).ToArray(),int.Parse(options.GetValueOrDefault("--resamples","1000"),System.Globalization.CultureInfo.InvariantCulture),ulong.Parse(options.GetValueOrDefault("--seed","1729"),System.Globalization.CultureInfo.InvariantCulture));
    var report=new ScoreReport(normalizer,"1",Sum(rows.Select(r=>r.Wer)),Sum(rows.Select(r=>r.Cer)),second is null?null:Sum(rows.Select(r=>r.SecondWer!)),bootstrap,rows);
    string output=options.GetValueOrDefault("--out","score.json");File.WriteAllText(output,JsonSerializer.Serialize(report,EvalJsonContext.Default.ScoreReport));
    string Percent(double? v)=>v is null?"undefined":(100*v.Value).ToString("F3",System.Globalization.CultureInfo.InvariantCulture)+"%";
    string CI(Interval? ci)=>ci is null?"unavailable":$"[{Percent(ci.Lower)}, {Percent(ci.Upper)}]";
    Console.WriteLine($"WER {Percent(report.Wer.Rate)} CI {CI(bootstrap.First)}; CER {Percent(report.Cer.Rate)}");
    if(report.SecondWer is {} other)Console.WriteLine($"WER2 {Percent(other.Rate)} CI {CI(bootstrap.Second)}; delta (first - second) {Percent(report.Wer.Rate-other.Rate)} CI {CI(bootstrap.Difference)}; {bootstrap.Verdict}");
    Console.WriteLine("Result: "+output);return 0;
}
catch(Exception ex) when(ex is ArgumentException or IOException or JsonException or FormatException or OverflowException or OneSpeech.InvalidTranscriptJsonException)
{Console.Error.WriteLine(ex.Message);return 2;}
