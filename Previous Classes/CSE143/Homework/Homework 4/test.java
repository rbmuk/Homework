public class test {
	public static void main(String[] args) {
		challenge(43269);
	}
	
	public static void challenge(int n) {
		System.out.print("+");
		if (n >= 10) {
			challenge(n/10);
		}
		if (n%2==0) {
			System.out.print("-");
		}
		else {
			System.out.print("*");
		}
	}
}