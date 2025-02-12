import java.util.*;

public class lecture18_notes {
	
	public static void main(String[] args) {
		
	}
	
	public static void stream() {
		List words = Arrays.asList("four", "score", "and", "seven", "years", "ago");
		words.stream().forEach(s -> System.out.println(s + " "));
		System.out.println();
		words.stream().forEach(System.out::print);
		System.out.println();
		words.stream().map(s -> s + " ").forEach(System.out::print);
		// Convention:
		words.stream()
			.map(s -> s + " ")
			.forEach(System.out::print);
	}
	
	public static void numbers() {
		List numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9);
		int sum1 = Arrays.stream(numbers).sum();
		
		// what about evens?
		int sum2 = Arrays.stream(numbers)
			.filter(n -> n % 2 == 0)
			.sum();
		System.out.println("sum2 : " + sum2);
		
		// Only evens, and adding up only absolute value
		int sum3 = Arrays.stream(numbers)
			.filter(n -> n % 2 == 0)
			.map(Math::abs)
			.sum();
		System.out.println("sum3 : " + sum3);
		
		// what about with no duplicates?
		int sum4 = Arrays.stream(numbers)
			.filter(n -> n % 2 == 0)
			.map(Math::abs)
			.distinct()
			.sum();
		System.out.println("sum4 : " + sum4);
		
		// summary of all built in stats
		System.out.println(Arrays.stream(numbers).summaryStatistics());
	}
	
	public static void rangeExample() {
		// prints 1 to 10
		IntStream.range(1, 11)
			.forEach(System.out::println);
		System.out.println();
		
		// print factorials of 0 through 10
		IntStream.range(0, 11)
			.map(lecture18_notes::factorial)
			.forEach(System.out::println);
		System.out.println();
		
		// print the primes between 1 and 23 inclusive
		IntStream.range(1, 24)
			.filter(isPrime(n))
			.forEach(System.out::println);
	}
	
	public static int factorial() {
		return IntStream.range(2, n+1)
			.reduce(1, (a, b) -> a*b); // 1 is the default value even if the stream is empty
	}
	
	public static void primesExample() {
		printSum(IntStream.range(1, 20001));
		System.out.println();
		
		printSum(IntStream.range(1, 20001).parallel()); // would run 16x faster!
		System.out.println();
	}
	
	public static boolean isPrime(int n) {
		return IntStream.range(1, n+1)
			.filter(m -> n % m == 0)
			.count()
			== 2;
	}
	
	public static void funnyExample() {
		IntStream.range(1, 21)
			.parallel()
			.forEach(System.out::println);
		System.out.println();
	}	
	
	public void sorting() {
		Arrays.sort(words, (a, b) -> a.length() - b.length());
	}
}