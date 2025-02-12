import java.util.*;

public class DayUnknown {
	public static void main(String[] args) {
		Queue<Character> a = new LinkedList<>();
		a.addAll(Arrays.asList('h', 'e', 'l', 'l', 'o'));
		printPerms("hello", a, "");
	}
	
	public static void printPerms(String s, Queue<Character> a, String result) {
		if (result.length() == s.length()) System.out.println(result);
		for (int i = 0; i < a.size(); ++i) {
			char c = a.remove();
			result += c;
			printPerms(s, a, result);
			a.add(c);
		}
	}
	
	public static void printDecimal(int n, String result) {
		if (n == 0) System.out.println(result);
		else
			for (int i = 0; i <= 9; ++i) 
				printDecimal(n-1, result+i);
	}
	
	public static void alternatingReverse(Stack<Integer> s) {
		Queue<Integer> q = new LinkedList<>();
		while (!s.isEmpty()) q.add(s.pop());
		int n = q.size();
		for (int i = 0; i < n; ++i) {
			if (i%2==1) q.add(q.remove());
			else s.push(q.remove());
		}
		for (int i = 0; i < n; ++i) {
			if (i%2==0) q.add(q.remove());
			else q.add(s.pop());
		}
		while (!q.isEmpty()) s.push(q.remove());
	}
	
	public static int evens(int n) {
		if (n < 0) return -evens(-n);
		if (n == 0) return 0;
		if (n < 10) return (n%2 == 0) ? n : 0;
		int last = evens(n%10);
		if (last != 0) return 10*evens(n/10)+last;
		else return evens(n/10);
	}
	
	public static int doubleDigit(int n, int d) {
		if (d < 0 || d > 10) throw new IllegalArgumentException();
		if (n < 0) return -doubleDigit(-n, d); 
		if (n < 10) return (n == d) ? n * 11 : n;
		int last = doubleDigit(n%10, d);
		if (last == 11*d) return 100*doubleDigit(n/10, d) + last;
		else return 10*doubleDigit(n/10, d) + last;
	}
	
	public static int digitProduct(int n) {
		if (n == 0) throw new IllegalArgumentException();
		if (n < 0) return -1 * digitProduct(-1 * n);
		if (n % 10 == 0) return digitProduct(n/10);
		if (n < 10) return n;
		return n%10 * digitProduct(n/10);
	}
	
	public static int digitMatch(int n, int d) {
		if (n < 0 || d < 0) throw new IllegalArgumentException();
		if (d < 10) return (n%10==d) ? 1 : 0;
		return digitMatch(n%10, d%10) + digitMatch(n/10, d/10);
	}
	
	public static String dedup(String s) {
		if (s.length() == 0) throw new IllegalArgumentException();
		if (s.length() == 1) return s;
		if (s.charAt(0) == s.charAt(1)) return dedup(s.substring(1));
		else return s.charAt(0) + dedup(s.substring(1));
	}
	
	public static void printSequence(int n) {
		if (n == 1) {
			System.out.print("*");
		} else if (n == 2) {
			System.out.print("**");
		} else if (n % 4 == 3 || n % 4 == 0) {
			System.out.print("<");
			printSequence(n-2);
			System.out.print(">");
		} else if (n % 4 == 1 || n % 4 == 2) {
			System.out.print(">");
			printSequence(n-2);
			System.out.print("<");
		}
	}
		
	public static void printRange(int s, int e) {
		if (s == e) {
			System.out.print(s);
		} else if (s == e-1) {
			System.out.print(s + " - " + e);
		} else {
			System.out.print(s + " < ");
			printRange(s+1, e-1);
			System.out.print(" > " + e);
		}
	}
	
	/*public void removeLast(int n) {
		int index = -1;
		for (int i = 0; i < size; ++i)
			if (n == elementData[i]) index = i;
		if (index == -1) return;
		for (int i = index; i < size - 1; ++i) {
			elementData[i] = elementData[i+1];
		}
		--size;
	}
	
	public static void mystery(int n) {
		System.out.print("+");
		if (n>=10) {
			mystery(n/10);
		}
		System.out.print((n%2==0)?"-":"*");
	}
	
	public static int doubleDigit(int n, int d) {
		if (d < 0 || d >= 10) throw new IllegalArgumentException();
		if (n < 0) {
			return -1 * doubleDigit(-1*n, d);
		}
		if (n<10) {
			if (n == d) {
				return 11*n;
			} else return n;
		} else {
			int z = doubleDigit(n%10, d);
			if (z > 10) {
				return z + 100*doubleDigit(n/10, d);
			} else return z + 10*doubleDigit(n/10, d);
		}
	}
	
/*	public ArrayIntList extractOddIndexes() {
		ArrayIntList a = new ArrayIntList();
		if (size == 0) return a;
		for (int i = 1; i < size; i += 2) {
			a.elementData[i] = elementData[i];
		}
		for (int i = 2; i < size; i += 2) {
			a.elementData[i/2] = a.elementData[i];
		}
		a.size = size/2;
		size = (int) Math.ceil(((double)size)/2);
		return a;
	}
	
	public static void mirrorSplit(Stack<Integer> s) {
		if (s.size() == 0) return;
		Queue<Integer> q = new LinkedList<>();
		while (!s.isEmpty()) {
			q.add(s.pop());
		}
		int n = q.size();
		for (int i = 0; i < n; ++i) {
			int b = q.remove();
			s.push((b%2==1)?b/2+1 : b/2);
			q.add(b/2);
		}
		while (!s.isEmpty()) {
			q.add(s.pop());
		}
		for (int i = 0; i < n; ++i) {
			q.add(q.remove());
		}
		
		while (!q.isEmpty()) {
			s.push(q.remove());
		}
	}
	
	public static void mystery5(int n) {
		System.out.print("*");
		if (n == 0) {
			System.out.print("=");
		} else {
			int y = n % 10;
			if (y < 5) {
				System.out.print(y);
				mystery5(n/10);
			} else {
				mystery5(n/10);
				System.out.print(y);
			}
		}
	}*/
}

class LinkedNode {
	public int value;
	public LinkedNode next;
	
	public LinkedNode(int value, LinkedNode next) {
		this.value = value;
		this.next = next;
	}
	
	public LinkedNode(int value) {
		this.value = value;
		this.next = null;
	}
}