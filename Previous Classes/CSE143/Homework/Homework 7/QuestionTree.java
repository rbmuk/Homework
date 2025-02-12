import java.util.*;
import java.io.*;

// This class constructs a Tree of Questions that 
// the user can answer yes/no to and will eventually 
// converge down to an answer, which will do different
// things based on if the answer is correct or incorrect.
public class QuestionTree {

	private Scanner console;
	private QuestionNode overallRoot;

	// Constructs a QuestionTree with one answer node
	// computer, and initially no question nodes.
	public QuestionTree() {
		console = new Scanner(System.in);
		overallRoot = new QuestionNode("computer");
	}
	
	// params: Scanner input: the scanner containing the tree
	// post: reads in a preivous tree specified by 
	// the scanner, which is in BNF format,
	// replacing the current tree.
	public void read(Scanner input) {
		overallRoot = read(input, overallRoot);
	}
	
	// post: at each step, will return "root" replaced with 
	// a new QuestionNode specified by a line in the scanner,
	// and if the type of the questionnode is a question ("Q"),
	// will also change root's left and right children,
	// eventually stopping when the next node is an answer or
	// if input has no more lines.
	private QuestionNode read(Scanner input, QuestionNode root) {
		if (!input.hasNext()) return null;
		String type = input.nextLine();
		root = new QuestionNode(input.nextLine());
		if (type.equals("Q:")) {
			root.left = read(input, root.left);
			root.right = read(input, root.right);
		}
		return root;
	}
	
	// params: PrintStream output: the printstream to be written to
	// post: writes the current tree to the printstream
	// in BNF format.
	public void write(PrintStream output) {
		write(output, overallRoot);
	}
	
	// params: the printstream to write to, and the current QuestionNode root
	// performs a pre-order write, in this form:
	// if root is an answer, will write "A \ (the answer)",
	// where \ represents a new line, else it will write "Q \ (the question)",
	// and then write the root's left/right children as well.
	private void write(PrintStream output, QuestionNode root) {
		if (root == null) return;
		output.println((root.isAnswer()) ? "A:" : "Q:");
		output.println(root.data);
		write(output, root.left);
		write(output, root.right);
	}
	
	// post: asks all the questions, and eventually gives an answer,
	// which, if correct, does nothing, else asks the user for the correct
	// answer and a question to distinguish it from the tree's answer,
	// and if the user would answer yes to the new question,
	// and adds the question/answer to the current tree.
	public void askQuestions() {
		overallRoot = askQuestions(overallRoot);
	}
	
	// post: 2 cases:
	// if root is an answer it will ask the user if it got the answer right,
	// do nothing if it did, else it will ask the user for the right answer and a 
	// question to distinguish the user's answer form the answer it gave,
	// ask if the answer to the given question should be yes/no,
	// and then add the question/answer pair to the correct place. 
	// If root is a question, and if the user answers "yes",
	// to the question, it will call this method on the left side of the tree,
	// else it calls this method on the right side of the tree.
	private QuestionNode askQuestions(QuestionNode root) {
		if (root.isAnswer()) {
			boolean b = yesTo("Would your answer happen to be " + root.data + "?");
			if (!b) {
				System.out.print("What is the name of your object? ");
				String answer = console.nextLine();
				System.out.println("Please give me a yes/no question that");
				System.out.println("distinguishes between your object");
				System.out.print("and mine--> ");
				String question = console.nextLine();
				boolean y = yesTo("And what is the answer for your object?");
				QuestionNode a = new QuestionNode(answer);
				if (y) return new QuestionNode(question, a, root);
				else return new QuestionNode(question, root, a);
			}
			System.out.println("Great, I got it right!");
			return root;
		}
		boolean b = yesTo(root.data);
		if (b) root.left = askQuestions(root.left);
		else root.right = askQuestions(root.right);
		return root;
	}
	
	// post: asks the user a question, forcing an answer of "y " or "n";
	// returns true if the answer was yes, returns false otherwise
	public boolean yesTo(String prompt) {
	 System.out.print(prompt + " (y/n)? ");
	 String response = console.nextLine().trim().toLowerCase();
	 while (!response.equals("y") && !response.equals("n")) {
	 System.out.println("Please answer y or n.");
	 System.out.print(prompt + " (y/n)? ");
	 response = console.nextLine().trim().toLowerCase();
	 }
	 return response.equals("y");
	}
}