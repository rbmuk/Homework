/*  Rohan Mukherjee
    TA: Mia Yamada Heidner

    The LetterInventory class keeps track of how many english letters
    are in a given String. */

public class LetterInventory {

    private int[] letterCount;
    private int size;
    public static final int NUMBER_OF_LETTERS = 26;

    /* post: creates a LetterInventory that is used
       to count the number of english letters in data */
    public LetterInventory(String data) {
        letterCount = new int[NUMBER_OF_LETTERS];
        size = 0;
        data = data.toLowerCase();
        for (int i = 0; i < data.length(); ++i) {
            if (!notALetter(data.charAt(i))) {
                letterCount[data.charAt(i) - 'a']++;
                size++;
            }
        }
    }

    /* pre: letter is a (lowercase or uppercase) letter
       if letter is not an english letter, throws an IllegalArgumentException */
    // post: returns the number of letter in the LetterInventory
    public int get(char letter) {
        if (notALetter(letter)) throw new IllegalArgumentException();
        letter = Character.toLowerCase(letter);
        return letterCount[letter - 'a'];
    }
    
    // post: returns the size of the LetterInventory
    public int size() {
        return size;
    }

    // post: returns true if LetterInventory is empty
    public boolean isEmpty() {
        return size == 0;
    }

    /* pre: letter is a (lowercase or uppercase) english letter, value is >= 0
       if letter is not an english letter, or if value < 0, 
       throws an IllegalArgumentException */
    /* post: changes the number of letter in the LetterInventory
       to value, and updates the size of LetterInventory */
    public void set(char letter, int value) {
        if (value < 0 || notALetter(letter)) throw new IllegalArgumentException();
        letter = Character.toLowerCase(letter);
        int currentSize = letterCount[letter - 'a'];
        letterCount[letter - 'a'] = value;
        size += value - currentSize;
    }

    // pre: nothing
    // post: returns LetterInventory in a readable form
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        for (int i = 0; i < NUMBER_OF_LETTERS; ++i) {
            for (int j = 0; j < letterCount[i]; ++j) {
                sb.append((char)('a' + i));
            }
        }
        sb.append(']');
        return sb.toString();
    }

    // pre: nothing
    /* post: returns the LetterInventory that is created
       by adding the number of letters component-wise
       (i.e. if we had "aa" + "a", we would get "aaa") */
    LetterInventory add(LetterInventory other) {
        int[] newLetterCount = new int[NUMBER_OF_LETTERS];
        size = 0;
        for (int i = 0; i < NUMBER_OF_LETTERS; ++i) {
            newLetterCount[i] = other.letterCount[i] + this.letterCount[i];
            size += newLetterCount[i];
        }
        LetterInventory added = new LetterInventory("");
        added.letterCount = newLetterCount;
        added.size = size;
        return added;
    }

    // pre: nothing
    /* post: returns the LetterInventory that is created
       by subtracting the number of letters component-wise
       (i.e. if we had "aa" - "a", we would get "a") */
    public LetterInventory subtract(LetterInventory other) {
        int[] newLetterCount = new int[NUMBER_OF_LETTERS];
        size = 0;
        for (int i = 0; i < NUMBER_OF_LETTERS; ++i) {
            int newCount = this.letterCount[i] - other.letterCount[i];
            if (newCount < 0) return null;
            newLetterCount[i] = newCount;
            size += newCount;
        }
        LetterInventory subtracted = new LetterInventory("");
        subtracted.letterCount = newLetterCount;
        subtracted.size = size;
        return subtracted;
    }

    private boolean notALetter(char c) {
        return c < 'A' || (c >= '[' && c < 'a') || c >= 'a' + NUMBER_OF_LETTERS;
    }
}