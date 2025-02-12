import java.util.*;
import java.io.*;

public class DayUnknown3 {
	public static void main(String[] args) throws FileNotFoundException {
		LinkedList l = new LinkedList(Arrays.asList(6, 7, 8, 12, 15, 20, 25, 42, 51));
		l.insertMultiplesOfN(6);
		//IntTree t = new IntTree(new Scanner(new File("tree.txt")));
		//System.out.println(t.countNodes(2, 1));
	}
}

class LinkedList {
	public ListNode front;
	public LinkedList(List<Integer> l) {
		front = new ListNode(l.get(0));
		ListNode t = front;
		for (int i = 1; i < l.size(); ++i) {
			t.next = new ListNode(l.get(i));
			t = t.next;
		}
	}
	
	public void insertMultiplesOfN(int N) {
		if (front == null) front = new ListNode(0);
		else {
			if (front.data != 0) front = new ListNode(0, front);
			ListNode t = front;
			while (t.next != null) {
				while (t.data/N * N + N < t.next.data) {
					t.next = new ListNode(t.data / N * N + N, t.next);
					t = t.next;
				}
				t = t.next;
			}
			if (t.data % N != 0) t.next = new ListNode(t.data / N * N + N);
		}
	}
	
	public boolean hasTwoConsectutive() {
		if (front != null) {
			ListNode t = front;
			while (t.next != null) {
				if (t.data == t.next.data - 1) return true;
				t = t.next;
			}
		}
		return false;
	}
	
	public void compressDuplicates() {
		if (front != null) {
			ListNode q = front;
			int count = 1;
			while (q.next != null && q.next.data == q.data) {
				count++;
				q = q.next;
			}
			ListNode l = new ListNode(count, q);
			front = l;
			ListNode p = q;
			q = q.next;
			while (q != null) {
				count = 1;
				while (q.next != null && q.next.data == q.data) {
					q = q.next;
					count++;
				}
				l = new ListNode(count, q);
				p.next = l;
				p = q;
				q = q.next;
			}
		}
	}
	
	public void printEvenOddSum() {
		int i = 0, even = 0, odd = 0;
		ListNode q = front;
		while (q != null) {
			if (i % 2 == 0) even += q.data;
			else odd += q.data;
			q = q.next;
			++i;
		}
		System.out.println("even sum = " + even);
		System.out.println("odd sum = " + odd);
	}
	
	public void removeAllNegatives() {
		while (front != null && front.data < 0)
			front = front.next;
		ListNode q = front;
		while (q != null && q.next != null) {
			if (q.next.data < 0) q.next = q.next.next;
			else q = q.next;
		}
	}
}

class ListNode {
	public int data;
	public ListNode next;
	public ListNode(int data, ListNode next) {
		this.data = data;
		this.next = next;
	}
	public ListNode(int data) { this(data, null); }
}

class IntTree {
	
	private IntTreeNode overallRoot;
	
	public IntTree(Scanner s) {
		overallRoot = new IntTreeNode(Integer.parseInt(s.nextLine()));
		overallRoot = cons(s, overallRoot);
	}
	
	private IntTreeNode cons(Scanner s, IntTreeNode root) {
		if (!s.hasNext() || root == null) return root;
		while (s.hasNextLine()) {
			List<String> l = Arrays.asList(s.nextLine().split(" "));
			if (l.size() == 1) return root;
			if (l.get(1).equals("L")) {
				root.left = new IntTreeNode(Integer.parseInt(l.get(0)));
				root.left = cons(s, root.left);
			} else {
				root.right = new IntTreeNode(Integer.parseInt(l.get(0)));
				root.right = cons(s, root.right);
			}
		}
		return root;
	}
	
	public IntTree(IntTreeNode o) {
		overallRoot = o;
	}
	
	public int countNodes(int x, int y) {
		if (x < 0 || y < 0) throw new IllegalArgumentException();
		return countNodes(overallRoot, x, y);
	}
	
	private int countNodes(IntTreeNode root, int x, int y) {
		if (root == null) return 0;
		if (x == 0 && y == 0) return 1;
		int left = 0, right = 0;
		if (x > 0) left += countNodes(root.left, x-1, y);
		if (y > 0) right += countNodes(root.right, x, y-1);
		return left + right;
	}
	
	public boolean alternatingParity() {
		return ap(overallRoot);
	}
	
	private boolean ap(IntTreeNode root) {
		if (root == null || root.left == null || root.right == null) return true;
		int one = (root.data + root.left.data); int two = root.data + root.right.data;
		return ((root.data + root.left.data) % 2 == 1) && ((root.data + root.right.data) % 2 == 1) 
		&& ap(root.left) && ap(root.right);
	}
	
	public int countBelow(int n) {
		if (n <= 0) throw new IllegalArgumentException();
		return countBelow(n-1, overallRoot);
	}
	
	private int countBelow(int n, IntTreeNode root) {
		if (root == null) return 0;
		return ((n <= 0) ? 1 : 0) + countBelow(n-1, root.left) + countBelow(n-1, root.right);
	}
	
	public void fill() {
		overallRoot = fill(overallRoot, height(overallRoot), 0);
	}
	
	private IntTreeNode fill(IntTreeNode root, int n, int m) {
		if (m < n) {
			if (root == null) root = new IntTreeNode(0);
			root.left = fill(root.left, n, m+1);
			root.right = fill(root.right, n, m+1);
		}
		return root;
	}
	
	private int height(IntTreeNode root) {
		if (root == null) return 0;
		return Math.max(height(root.left), height(root.right)) + 1;
	}
}

class IntTreeNode {
	
	public int data;
	public IntTreeNode left, right;
	
	public IntTreeNode(int data, IntTreeNode left, IntTreeNode right) {
		this.data = data; this.left = left; this.right = right;
	}
	
	public IntTreeNode(int data) {
		this(data, null, null);
	}	
}