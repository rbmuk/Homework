import java.util.*;

public class Day3 {
	public static void main(String[] args) {
		Queue<Integer> q = new LinkedList<Integer>();
		q.add(3);
		q.add(2);
		q.add(1);
		System.out.println(q);
		reverse(q);
		System.out.println(q);
		/*Stack<Integer> s = new Stack<>();
		s.push(1);
		s.push(3);
		s.push(-5);
		s.push(-2342);
		s.push(7);
		System.out.println(s);
		splitStack(s);
		System.out.println(s);
		stutter(s);
		System.out.println(s);*/
	}
	
	public static void splitStack(Stack<Integer> s) {
		Queue<Integer> q = new LinkedList<>();
		while (!s.isEmpty()) {
			q.add(s.pop());
		}
		int size = q.size();
		for (int i = 0; i < size; ++i) {
			int n = q.remove();
			if (n < 0) s.push(n);
			else q.add(n);
		}
		while (!q.isEmpty()) s.push(q.remove());
	}
	
	public static void stutter(Stack<Integer> s) {
		Queue<Integer> q = new LinkedList<>();
		while (!s.isEmpty()) q.add(s.pop());
		
		// reverse queue
		while (!q.isEmpty()) s.push(q.remove());
		while (!s.isEmpty()) q.add(s.pop());
		
		// add back to stack
		while (!q.isEmpty()) {
			int n = q.remove();
			for (int j = 0; j < 2; ++j) s.push(n);
		}
	}
	
	private static boolean notALetter(char c) {
        return c < 'A' || (c >= 'Z' + 1 && c < 'a') || c >= 'a' + 26;
    }
	
	public static void reverse(Queue<Integer> q) {
		if (q.isEmpty()) return;
		int m = q.remove();
		reverse(q);
		q.add(m);
	}
	
	public static void isPalindrome(Queue<Integer> q) {
		boolean isOdd = (q.size % 2 == 1);
		int odd;
		Stack<Integer> s = new Stack<>();
		int n = q.size()/2;
		for (int i = 0; i < n; ++i) {
			s.push(q.remove());
		}
		if (isOdd) odd = q.remove();
		while (!q.isEmpty()) {
			if (q.remove() != s.pop()) return false;
		}
		return true;
	}
}