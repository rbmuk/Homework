import java.util.*;

public class Test {
	public static void main(String[] args) {
		List ki = Arrays.asList("Jim", "Bob", "Tim");
		AssassinManager am = new AssassinManager(ki);
		am.printKillRing();
		am.kill("BOB");
		System.out.println(am.gameOver());
		am.kill("Jim");
		System.out.println(am.graveyardContains("Bob"));
		System.out.println(am.graveyardContains("Tim"));
		am.printGraveyard();
		am.kill("Jimothy");
		System.out.println(am.gameOver());
		System.out.println(am.winner());
	}
}