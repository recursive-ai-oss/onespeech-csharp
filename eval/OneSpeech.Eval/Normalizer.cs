using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace OneSpeech.Evaluation;

// These are our scoring rules, not an implementation-compatible reference normalizer.
public static partial class Normalizer
{
    private static readonly Dictionary<string, string> Contractions = new(StringComparer.Ordinal)
    {
        ["can't"]="can not", ["cannot"]="can not", ["won't"]="will not", ["shan't"]="shall not",
        ["ain't"]="is not", ["let's"]="let us", ["it's"]="it is", ["that's"]="that is", ["what's"]="what is",
        ["there's"]="there is", ["here's"]="here is", ["he's"]="he is", ["she's"]="she is",
        ["gonna"]="going to", ["wanna"]="want to", ["gotta"]="got to", ["imma"]="i am going to"
    };
    private static readonly Dictionary<string,string> Spellings = new(StringComparer.Ordinal)
    {
        ["colour"]="color", ["colours"]="colors", ["favour"]="favor", ["favourite"]="favorite",
        ["honour"]="honor", ["labour"]="labor", ["neighbour"]="neighbor", ["centre"]="center",
        ["centres"]="centers", ["theatre"]="theater", ["metre"]="meter", ["litre"]="liter",
        ["organise"]="organize", ["organised"]="organized", ["recognise"]="recognize", ["realise"]="realize",
        ["travelling"]="traveling", ["cancelled"]="canceled", ["grey"]="gray", ["defence"]="defense", ["licence"]="license"
    };
    private static readonly string[] Small = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".Split(' ');
    private static readonly string[] Tens = "twenty thirty forty fifty sixty seventy eighty ninety".Split(' ');
    private static readonly Dictionary<string,int> Ordinals = new(StringComparer.Ordinal)
    { ["first"]=1,["second"]=2,["third"]=3,["fourth"]=4,["fifth"]=5,["sixth"]=6,["seventh"]=7,["eighth"]=8,["ninth"]=9,["tenth"]=10,["eleventh"]=11,["twelfth"]=12,["twentieth"]=20,["thirtieth"]=30,["hundredth"]=100 };

    public static string Normalize(string text, string mode)
    {
        if (mode is not ("en" or "basic")) throw new ArgumentException("Normalizer must be en or basic.");
        text=text.Normalize(NormalizationForm.FormKC);
        text=mode=="basic"?CaseFolding.Fold(text):text.ToLowerInvariant();
        text = Brackets().Replace(text, " ");
        if (mode == "en")
        {
            text = text.Replace('’', '\'').Replace('‘', '\'');
            text = BeforeApostrophe().Replace(text, "'");
            text = Words().Replace(text, m => Contractions.GetValueOrDefault(m.Value, Expand(m.Value)));
            text = Filler().Replace(text, " ");
            text = DigitComma().Replace(text, "");
            text = text.Normalize(NormalizationForm.FormD);
        }
        var b = new StringBuilder();
        foreach (var rune in text.EnumerateRunes())
        {
            var c = Rune.GetUnicodeCategory(rune);
            bool mark = c is UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.EnclosingMark;
            bool symbol = c is UnicodeCategory.MathSymbol or UnicodeCategory.CurrencySymbol or UnicodeCategory.ModifierSymbol or UnicodeCategory.OtherSymbol;
            bool punct = c is UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DashPunctuation or UnicodeCategory.OpenPunctuation or UnicodeCategory.ClosePunctuation or UnicodeCategory.InitialQuotePunctuation or UnicodeCategory.FinalQuotePunctuation or UnicodeCategory.OtherPunctuation;
            if (mode == "en" && (rune.Value is '.' or '%' or '$' or '£' or '€')) b.Append(rune);
            else if (mark || symbol || punct) { if (!(mode == "en" && mark)) b.Append(' '); }
            else b.Append(rune);
        }
        var tokens = Space().Split(b.ToString().Trim()).Where(x => x.Length > 0).ToArray();
        if (mode == "basic") return string.Join(' ', tokens);
        var result = new List<string>();
        for (int i=0;i<tokens.Length;i++)
        {
            string token = tokens[i];
            if (TryNumber(tokens, ref i, out string number)) token = number;
            if (i+1 < tokens.Length && tokens[i+1] is "dollars" or "dollar" or "pounds" or "pound" or "euros" or "euro" && Numeric().IsMatch(token))
            { token = (tokens[++i].StartsWith("dollar", StringComparison.Ordinal) ? "$" : tokens[i].StartsWith("pound", StringComparison.Ordinal) ? "£" : "€") + token; }
            else if (i+1<tokens.Length && tokens[i+1] is "percent" && Numeric().IsMatch(token)) { token += "%"; i++; }
            token = Spellings.GetValueOrDefault(token, token);
            token = Leftover().Replace(token, " ");
            if (token.Length>0) result.Add(token);
        }
        return Space().Replace(string.Join(' ', result), " ").Trim();
    }
    private static string Expand(string word)
    {
        foreach (var (suffix, expansion) in new[] { ("n't"," not"),("'re"," are"),("'ve"," have"),("'ll"," will"),("'m"," am"),("'d"," would") })
            if (word.EndsWith(suffix, StringComparison.Ordinal)) return word[..^suffix.Length]+expansion;
        return word;
    }
    private static bool TryNumber(string[] words, ref int index, out string output)
    {
        output=""; decimal total=0, group=0; int j=index; bool seen=false, ordinal=false; string prefix="";
        for (;j<words.Length;j++)
        {
            string w=words[j];
            if (!seen)
            {
                string numeric=w;
                if(w.Length>1 && w[0] is '$' or '£' or '€') {prefix=w[..1];numeric=w[1..];}
                if(decimal.TryParse(numeric,NumberStyles.AllowDecimalPoint,CultureInfo.InvariantCulture,out decimal literal)) {group=literal;seen=true;continue;}
            }
            int n=Array.IndexOf(Small,w);
            if (n>=0) { if(seen && group%10!=0) break; group+=n; seen=true; continue; }
            n=Array.IndexOf(Tens,w);
            if (n>=0) { group+=(n+2)*10; seen=true; continue; }
            if (Ordinals.TryGetValue(w,out n)) { group+=n; seen=true; ordinal=true; j++; break; }
            if (w=="hundred") { group=Math.Max(1,group)*100; seen=true; continue; }
            decimal scale=w switch { "thousand"=>1000,"million"=>1000000,"billion"=>1000000000,_=>0 };
            if (scale>0) { total+=Math.Max(1,group)*scale; group=0; seen=true; continue; }
            if (seen && w=="and" && j+1<words.Length && (Array.IndexOf(Small,words[j+1])>=0 || Array.IndexOf(Tens,words[j+1])>=0)) continue;
            break;
        }
        if (!seen) return false;
        decimal value=total+group;
        output=prefix+value.ToString(CultureInfo.InvariantCulture);
        if (!ordinal && j<words.Length && words[j]=="point")
        {
            int k=j+1; string fraction="";
            while(k<words.Length && Array.IndexOf(Small,words[k]) is >=0 and <=9) fraction+=Array.IndexOf(Small,words[k++]).ToString(CultureInfo.InvariantCulture);
            if(fraction.Length>0) { output+="."+fraction; j=k; }
        }
        if(ordinal) { int v=(int)(value%100); output+=v is >=11 and <=13 ? "th" : (v%10) switch {1=>"st",2=>"nd",3=>"rd",_=>"th"}; }
        index=j-1; return true;
    }
    [GeneratedRegex(@"\[[^\[\]]*\]|\([^()]*\)")] private static partial Regex Brackets();
    [GeneratedRegex(@"\s+’|\s+'")] private static partial Regex BeforeApostrophe();
    [GeneratedRegex(@"[a-z]+(?:'[a-z]+)?")] private static partial Regex Words();
    [GeneratedRegex(@"\b(hmm|mm|mhm|mmm|uh|um)\b")] private static partial Regex Filler();
    [GeneratedRegex(@"(?<=\d),(?=\d)")] private static partial Regex DigitComma();
    [GeneratedRegex(@"\s+")] private static partial Regex Space();
    [GeneratedRegex(@"^\d+(\.\d+)?$")] private static partial Regex Numeric();
    [GeneratedRegex(@"(?<!\d)\.|\.(?!\d)|[$£€](?!\d)|(?<!\d)%")] private static partial Regex Leftover();
}
