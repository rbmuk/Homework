import java.util.*;

// Rohan Mukherjee
// TA: Mia Yamada-Heidner
// CSE 143 Section A

// The AnagramSolver class is given a list of words to generate anagrams
// from, and the user is able to ask the class to generate thoes words.

public class AnagramSolver {
	
	private Map<String, LetterInventory> dictionary;
	private LetterInventory letterInventory;
	
	// params: list: the dictionary of words to generate anagrams from
	// post: initializes the anagramsolver with the given dictionary
	public AnagramSolver(List<String> list) {
		LetterInventory li = new LetterInventory(s);
		dictionary = new TreeMap<>();
		for (String s2 : list) {
			LetterInventory li2 = new LetterInventory(s2);
			if (letterInventory.subtract(li2) != null) {
				map.put(s2, li2);
			}
		}
	}
	
	// params: String s: the word to use to generate anagrams, and 
	// int max: the maximum number of words in an anagram
	// pre: max >= 0
	// post: prints out all the possible anagrams that can 
	// be generated from s and the list given in the constructor,
	// and will not print out anagrams with more words than max, except
	// when max = 0 where it will print out all possible anagrams (with
	// no limit on the number of words)
	public void print(String s, int max) {
		if (max < 0) throw new IllegalArgumentException();
		Stack<String> st = new Stack<String>();
		if (max == 0) {
			print(letterInventory, Integer.MAX_VALUE, st);
		} else {
			print(letterInventory, max, st);
		}
	}
	
	// params: LetterInventory li: the current LetterInventory to see
	// which words can be removed from it to make an anagram, 
	// int max: the max number of words in any anagram, and
	// Stack<String> s: the current anagram
	// post: prints out all the possible anagrams with <= max words, given
	// the possible letters specified by LetterInventory
	private void print(LetterInventory li, int max, Stack<String> st) {
		if (st.size() > max) return;
		if (li.isEmpty()) System.out.println(st);
		else {
			for (String s : dictionary.keySet()) {
				LetterInventory li2 = dictionary.get(s);
				if (li.subtract(li2) != null) {
					st.push(s);
					print(li.subtract(li2), max, st);
					st.pop();
				}
			}
		}
	}
}