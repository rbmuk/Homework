// The Dictionary class provides an interface for reading a standard dictionary
// file and answering basic questions about whether it contains certain words
// and prefixes.  It also provides a mechanism for viewing the words as an
// unmodifiable sorted list.  All words from the dictionary file are converted
// to lowercase.

import java.util.*;
import java.io.*;

public class Dictionary {
    private String[] words;

    public static final String DICTIONARY_FILE = "dictionary.txt";

    // pre : current directory contains the dictionary file
    // post: constructs a dictionary object using the standard dictionary file
    public Dictionary() {
        Scanner input;
        try {
            input = new Scanner(new File(DICTIONARY_FILE));
        } catch (FileNotFoundException e) {
            throw new RuntimeException("cannot find dictionary file");
        }
        Set<String> entries = new TreeSet<>();
        while (input.hasNext()) {
            entries.add(input.next().toLowerCase());
        }
        words = entries.toArray(new String[] {});
    }

    // post: returns whether or not the dictionary contains the given word
    //       ignoring case
    public boolean contains(String word) {
        return Arrays.binarySearch(words, word.toLowerCase()) >= 0;
    }

    // post: returns whether or not some word in the dictionary has the given
    //       prefix ignoring case
    public boolean containsPrefix(String prefix) {
        prefix = prefix.toLowerCase();
        // first find index where this prefix occurs or would be inserted
        int index = Arrays.binarySearch(words, prefix);
        // return whether the word at that spot actually is the given prefix or
        // whether some word has the given prefix
        return index >= 0 || words[-index - 1].startsWith(prefix);
    }

    // post: returns the number of words in the dictionary
    public int size() {
        return words.length;
    }

    // post: returns a list view of the dictionary as an unmodifiable list
    public List<String> wordList() {
        return Collections.unmodifiableList(Arrays.asList(words));
    }
}