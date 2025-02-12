// Program to print out combinations of strings that appear in a dictionary.
// The program makes use of a Dictionary object and prompts for a file name
// with the strings to consider for forming words.

import java.io.*;
import java.util.*;

public class Jumble {
    public static void main(String[] args) throws FileNotFoundException {
        // prompt for file name
        Scanner console = new Scanner(System.in);
        System.out.println("Welcome to the CSE143 Word Jumble Program");
        System.out.println();
        System.out.print("What file of strings do you want to use? ");
        String fileName = console.nextLine();

        // read strings into a list
        Scanner input = new Scanner(new File(fileName));
        List<String> options = new ArrayList<>();
        while (input.hasNext()) {
            options.add(input.next());
        }

        // construct the dictionary
        Dictionary words = new Dictionary();
        
        //System.out.println("Jumbles of length 3:");
        //printJumbles(words, options, 3);
        //System.out.println();

        System.out.println("Jumbles of any length:");
        printJumbles2(words, options);
        System.out.println();

        System.out.println("Jumbles of any length with no repeated strings:");
        printJumbles3(words, options);
    }

    // Prints all sequences of strings chosen from options and using the given
    // number of words that are words in the dictionary.
   public static void printJumbles(Dictionary words, List<String> options, int choices) {
		printJumbles(words, options, choices, "");
	}
	
	// pre: dictionary has a word with prefix result
	private static void printJumbles(Dictionary words, List<String> options, int choices, String result) {
		if (choices == 0)
			if (words.contains(result)) System.out.println(result);
		else
			for (String s : options)
				printJumbles(words, options, choices - 1, result + s);
	}


    // Prints all sequences of strings chosen from options and that are words
    // in the dictionary.
    public static void printJumbles2(Dictionary words, List<String> options) {
        printJumbles2(words, options, "");
    }
	
	private static void printJumbles2(Dictionary words, List<String> options, String result) {
		if (words.contains(result)) System.out.println(result);
		for (String s : options)
			if (words.containsPrefix(result + s))
				printJumbles2(words, options, result + s);
	}


    // Prints all sequences of strings chosen from options and that are words
    // in the dictionary and using each string at most once.
    public static void printJumbles3(Dictionary words, List<String> options) {
        //to be completed for problem 3
		printJumbles3(words, options, "", new HashSet<String>());
    }
	
	private static void printJumbles3(Dictionary words, List<String> options, String result, Set<String> used) {
		if (words.contains(result)) System.out.println(result);
		for (String s : options)
			if (words.containsPrefix(result + s) && !used.contains(s))
			{
				used.add(s);
				printJumbles3(words, options, result + s, used);
				used.remove(s);
			}
	}
}