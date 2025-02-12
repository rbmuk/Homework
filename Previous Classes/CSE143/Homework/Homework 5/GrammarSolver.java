import java.util.*;

// Rohan Mukherjee
// TA: Mia Yamada-Heidner

// The GrammarSolver class takes in a list of grammars and 
// can be used to generate these symbols. 
public class GrammarSolver {
	
	private SortedMap<String, ArrayList<String>> symbols;
	private Random rand;

	// params: grammar: the list of grammar rules in BNF format
	// pre: grammar is not empty and there is only one set of rules
	// for each nonterminal else throws IllegalArgumentException
	// post: creates a GrammarSolver instance, using the grammar supplied by
	// "grammar"
	public GrammarSolver(List<String> grammar) {
		if (grammar.size() == 0) throw new IllegalArgumentException();
		symbols = new TreeMap<>();
		for (String s : grammar) {
			String[] line = s.split("::=");
			String nonterminal = line[0];
			if (symbols.keySet().contains(nonterminal))
				throw new IllegalArgumentException();
			ArrayList<String> rules = new ArrayList<>();
			String[] rulesAr = line[1].split("[|]");
			for (String st : rulesAr)
				rules.add(st.trim());
			symbols.put(nonterminal, rules);
		}
		rand = new Random();
	}
	
	// returns true if the "symbol" is a nonterminal
	public boolean grammarContains(String symbol) {
		return symbols.keySet().contains(symbol);
	}
	
	// params: symbol: the symbol to be generated, times: the number of times to do so
	// pre: grammarContains(symbol) and times >= 0, else throws IllegalArgumentException
	// post: returns a String[] of size times with the desired symbol
	public String[] generate(String symbol, int times) {
		if (!grammarContains(symbol) || times < 0) throw new IllegalArgumentException();
		String[] ret = new String[times];
		for (int i = 0; i < times; ++i)
			ret[i] = generate(symbol).trim();
		return ret;
	}
	
	// params: symbol: the symbol to be generated
	// post: returns the generated "symbol"
	private String generate(String symbol) {
		if (!grammarContains(symbol)) {
			return symbol + " ";
		} else {
			String choice = symbols.get(symbol).get(rand.nextInt(symbols.get(symbol).size()));
			String[] parts = choice.split("[ \t]+");
			String ret = "";
			for (int i = 0; i < parts.length; ++i)
				ret += generate(parts[i]);
			return ret;
		}
	}
	
	// returns the symbols in a readable form, in sorted order
	public String getSymbols() {
		return symbols.keySet().toString();
	}
}