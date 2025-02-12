// This is the question node that will be used by question tree.
public class QuestionNode {
	
	public String data;
	public QuestionNode left;
	public QuestionNode right;
	private final boolean isAnswer;
	
	// params: String data: the question OR answer,
	// QuestionNode left: the left question node, and QuestionNode right:
	// the right question node
	// post: Creates a QuestionNode with the data specified above
	public QuestionNode(String data, QuestionNode left, QuestionNode right) {
		isAnswer = (left == null && right == null);
		this.data = data;
		this.left = left;
		this.right = right;
	}
	
	// params: String s
	// post: creates a question node with no left/right  child, and 
	// data = s.
	public QuestionNode(String s) {
		this(s, null, null);
	}
	
	// returns true if the current node is a question node--never changes
	public boolean isAnswer() {
		return isAnswer;
	}
}