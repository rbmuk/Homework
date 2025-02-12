import java.util.*;

// Rohan Mukherjee
// TA: Mia Yamada-Heidner
// CSE 143 Section A

// The HangmanManager class manages all the important info 
// about a game of hangman. It picks the word as late as possible,
// to make the game harder. It keeps track of the letters guessed,
// and can show what the current guess "looks" like.
public class HangmanManager {

	private Set<String> words;
	private int guessesLeft;
	private Set<Character> guesses;
	private String currentPattern;

	// params: dictionary: the list of words guessable
	// 		   length: the length of word that will be used
	// 		   max: the max number of guesses allowed
	// pre: length >= 1, max >= 0, else throws IllegalArgumentException
	// post: Creates the HangmanManager with only words that have length
	// "length", and sets the number of guesses to "max".
	public HangmanManager(Collection<String> dictionary, int length, int max) {
		if (length < 1 || max < 0) throw new IllegalArgumentException();
		words = new TreeSet<>();
		for (String word : dictionary) {
			if (word.length() == length)
				words.add(word);
		}
		guessesLeft = max;
		guesses = new TreeSet<>();
		currentPattern = "";
		currentPattern += "-";
		for (int i = 0; i < length - 1; ++i) {
			currentPattern += " -";
		}
	}
	
	// returns the set of remaining words
	public Set<String> words() {
		return words;
	}
	
	// returns the number of guesses left
	public int guessesLeft() {
		return guessesLeft;
	}
	
	// returns a set of the guesses guessed
	public Set<Character> guesses() {
		return guesses;
	}
	
	// pre: the set of words is not empty, else throws IllegalStateException()
	// post: returns the pattern associated with the current guesses.
	// for example, if the letters guessed were {a, b, c, d}
	// and the word was ball, it would output "b a - -".
	public String pattern() {
		if (words.size() == 0) throw new IllegalStateException();
		return currentPattern;
	}
	
	// params: guess: the guess to be guessed
	// pre: the set of words is not empty and you have at least
	// 1 guess left, else throws an IllegalStateException()
	// post: decides what the most evil set of words to use is, uses it, 
	// returns the number of "guess" in the new pattern, and takes away
	// a guess if the user's guess didn't gain any new letters from "guess".
	public int record(char guess) {
		if (words.size() == 0 || guessesLeft < 1) throw new IllegalStateException();
		if (guesses.contains(guess)) throw new IllegalArgumentException();
		guesses.add(guess);
		Map<String, Set<String>> map = new TreeMap<>();
		for (String word : words) {
			add(map, word);
		}
		words = findLargestSet(map);
		String word = firstWordInSet(words);
		currentPattern = convert(word);
		int ret = numberOfLettersInString(word, guess);
		if (ret == 0) --guessesLeft;
		return ret;
	}
	
	// returns the number of times "c" shows up in "s"
	private int numberOfLettersInString(String s, char c) {
		int ret = 0;
		for (int i = 0; i < s.length(); ++i) {
			if (s.charAt(i) == c) ++ret;
		}
		return ret;
	}
	
	// this method returns the set of largest size in "map"
	private Set<String> findLargestSet(Map<String, Set<String>> map) {
		int maxLen = -1;
		Set<String> maxSet = new TreeSet<>();
		for (String key : map.keySet()) {
			if (map.get(key).size() > maxLen) {
				maxLen = map.get(key).size();
				maxSet = map.get(key);
			}
		}
		return maxSet;
	}
	
	// pre: s.length() >= 1
	// post: converts the word to the pattern
	// described in the method pattern()
	private String convert(String s) {
		String ret = "";
		if (guesses.contains(s.charAt(0))) ret += s.charAt(0);
		else ret += "-";
		for (int i = 1; i < s.length(); ++i) {
			ret += " ";
			if (guesses.contains(s.charAt(i))) ret += s.charAt(i);
			else ret += "-";
		}
		return ret;
	}
	
	// this method adds "word" to "map"
	private void add(Map<String, Set<String>> map, String word) {
		String converted = convert(word);
		if (!map.containsKey(converted)) {
			Set<String> ts = new TreeSet<>();
			map.put(converted, ts);
		}
		map.get(converted).add(word);
	}
	
	// this method returns a String from the set "set"
	private String firstWordInSet(Set<String> set) {
		return set.iterator().next();
	}
}