using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace OneSpeech.Evaluation;

public sealed record EvalWord(string Text,double? Start=null,double? End=null);
public sealed record Utterance(string Id,string Text,EvalWord[]? Words=null);
public sealed record UtteranceScore(string Id,ErrorCounts Wer,ErrorCounts Cer,ErrorCounts? SecondWer,BoundarySummary? Timing);
public sealed record ScoreReport(string Normalizer,string NormalizerVersion,ErrorCounts Wer,ErrorCounts Cer,ErrorCounts? SecondWer,BootstrapResult Bootstrap,UtteranceScore[] Utterances);

[JsonSourceGenerationOptions(PropertyNamingPolicy=JsonKnownNamingPolicy.CamelCase,WriteIndented=true)]
[JsonSerializable(typeof(ScoreReport))]
[JsonSerializable(typeof(Utterance))]
public partial class EvalJsonContext:JsonSerializerContext;

public static class Hypotheses
{
    public static Dictionary<string,Utterance> Read(string path)
    {
        var result=new Dictionary<string,Utterance>(StringComparer.Ordinal);
        // Transcript envelopes are single documents. JSONL records are independently bounded.
        using(var stream=File.OpenRead(path))
        {
            if(stream.Length<=8*1024*1024)
            {
                using var reader=new StreamReader(stream,new UTF8Encoding(false,true));string all=reader.ReadToEnd();
                try
                {
                    using var doc=JsonDocument.Parse(all,new JsonDocumentOptions{MaxDepth=32});
                    if(doc.RootElement.ValueKind==JsonValueKind.Object && doc.RootElement.TryGetProperty("schemaVersion",out _))
                    {
                        var t=OneSpeech.Transcript.FromJson(all);
                        var u=new Utterance(Path.GetFileNameWithoutExtension(path),t.Text,t.Words.Select(w=>new EvalWord(w.Text,w.Start?.TotalSeconds,w.End?.TotalSeconds)).ToArray());
                        result.Add(u.Id,u);return result;
                    }
                }
                catch(JsonException) { /* Multiple JSONL objects are parsed individually below. */ }
            }
        }
        using var input=new StreamReader(path,new UTF8Encoding(false,true));
        while(ReadBoundedLine(input) is {} line)
        {
            if(string.IsNullOrWhiteSpace(line))continue;
            using var doc=JsonDocument.Parse(line,new JsonDocumentOptions{MaxDepth=32});var root=doc.RootElement;
            if(root.ValueKind!=JsonValueKind.Object)throw new InvalidDataException("Expected a JSONL object.");
            if(root.TryGetProperty("type",out var type) && type.ValueKind==JsonValueKind.String && type.GetString()=="header")continue;
            var item=JsonSerializer.Deserialize(line,EvalJsonContext.Default.Utterance)??throw new InvalidDataException("Null utterance.");
            if(string.IsNullOrWhiteSpace(item.Id)||item.Text is null)throw new InvalidDataException("Each utterance needs id and text.");
            foreach(var w in item.Words??[])
                if(w is null || w.Text is null || (w.Start is null)!=(w.End is null) || w.Start is {} s && (!double.IsFinite(s)||s<0||w.End<s||!double.IsFinite(w.End!.Value)))throw new InvalidDataException("Invalid word interval.");
            if(!result.TryAdd(item.Id,item))throw new InvalidDataException("Duplicate utterance ID: "+item.Id);
        }
        return result;
    }
    private static string? ReadBoundedLine(StreamReader reader)
    {
        var b=new StringBuilder();int c;
        while((c=reader.Read())!=-1 && c!='\n') {if(b.Length>=2*1024*1024)throw new InvalidDataException("JSONL line exceeds 2 Mi characters.");b.Append((char)c);}
        return c==-1 && b.Length==0?null:b.ToString();
    }
    public static void Write(string path,IEnumerable<Utterance> records)
    {
        using var writer=new StreamWriter(path,false,new UTF8Encoding(false));
        foreach(var u in records) writer.WriteLine(JsonSerializer.Serialize(u,EvalJsonContext.Default.Utterance).Replace("\n", ""));
    }
    public static IEnumerable<Utterance> LibriSpeech(string directory)
    {
        foreach(string path in Directory.EnumerateFiles(directory,"*.trans.txt",SearchOption.AllDirectories).Order(StringComparer.Ordinal))
            foreach(string line in File.ReadLines(path))
            {int space=line.IndexOf(' ');if(space<=0)throw new InvalidDataException("Invalid LibriSpeech transcript line.");yield return new(line[..space],line[(space+1)..]);}
    }
    public static IEnumerable<Utterance> Earnings22(string directory)
    {
        foreach(string path in Directory.EnumerateFiles(directory,"*.nlp",SearchOption.AllDirectories).Order(StringComparer.Ordinal))
        {
            var words=new List<EvalWord>();bool header=true;
            foreach(string line in File.ReadLines(path))
            {
                if(header) {if(!line.StartsWith("token|speaker|ts|endTs|",StringComparison.Ordinal))throw new InvalidDataException("Invalid NLP header.");header=false;continue;}
                if(line.Length==0)continue;var fields=line.Split('|');if(fields.Length!=8)throw new InvalidDataException("Expected eight NLP fields.");
                double? Number(string s)=>string.IsNullOrEmpty(s)?null:double.Parse(s,CultureInfo.InvariantCulture);
                words.Add(new(fields[0],Number(fields[2]),Number(fields[3])));
            }
            if(header)throw new InvalidDataException("Missing NLP header.");
            yield return new(Path.GetFileNameWithoutExtension(path),string.Join(' ',words.Select(w=>w.Text)),words.ToArray());
        }
    }
}
