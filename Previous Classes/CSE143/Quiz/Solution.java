class Solution {
	public static void main(String[] args) {
		
	}
	
	public static void printJumbles(Dictionary words, List<String> options, int choices) {
		printJumbles2(words, options, choices, "");
	}
	
	// pre: dictionary has a word with prefix result
	private static void printJumbles2(Dictionary words, List<String> options, int choices, String result) {
		if (choices = 0)
			if (words.contains(result) System.out.println(result);
		else
			for (String s : options)
				if (words.containsPrefix(result + s))
					printJumbles(words, options, choices - 1, result + s);
	}
	
	private static void printJumbles2(Dictionary words, List<String> options, int choices, String result, Set<String> used) {
		if (choices = 0)
			if (words.contains(result) System.out.println(result);
		else
			for (String s : options)
				if (!used.contains(s))
					if (words.containsPrefix(result + s)) {
						used.add(s);
						printJumbles(words, options, choices - 1, result + s);
					}
	}