import java.util.*;

public class AssassinManager {

	private AssassinNode front;
	private AssassinNode graveyard;
	
	// params: names -- the list of names to be added to the kill ring
	// pre: names.length() > 0
	// else, throws IllegalArgumentException
	// post: creates a kill ring with the names specified in names, one after the other
	// with names.get(0) stalking names.get(1), ..., and names.get(names.length() - 1)
	// stalking names.get(0)
	public AssassinManager(List<String> names) {
		if (names.size() == 0) throw new IllegalArgumentException();
		front = new AssassinNode(names.get(0));
		AssassinNode temp = front;
		for (int i = 1; i < names.size(); ++i) {
			temp.next = new AssassinNode(names.get(i));
			temp = temp.next;
		}
	}
	
	// post: prints the kill ring in a neat fashion--e.g.
	// if the kill ring is Jim->Bob->Tim then it will print
	//     Jim is stalking Bob
	//     Bob is stalking Tim
	//     Tim is stalking Jim
	// with the last person stalking the first person, and four spaces
	// before every outputted line.
	// if there is only one person in the ring, named name, it will print
	//     name is stalking name
	public void printKillRing() {
		AssassinNode temp = front;
		
		// front + middle
		while (temp.next != null) {
			System.out.println("    " + temp.name + " is stalking " + temp.next.name);
			temp = temp.next;
		}
		
		// End
		System.out.println("    " + temp.name + " is stalking " + front.name);
	}
	
	// will print the names of the people killed, from most recent to
	// least recent, in the following way:
	// if name is the name of the person killed, and killer is the name
	// of the person who killed name, it will print
	//     name was killed by killer
	// with 4 spaces before the word "name". It will produce no output
	// if the graveyard is empty.
	public void printGraveyard() {
		AssassinNode temp = graveyard;
		while (temp != null) {
			System.out.println("    " + temp.name + " was killed by " + temp.killer);
			temp = temp.next;
		}
	}
	
	// ignores case
	// params: name--name of the person to check if they are in the kill ring
	// post: returns true if "name" is currently in the kill ring
	public boolean killRingContains(String name) {
		AssassinNode temp = front;
		boolean contains = false;
		while (temp != null) {
			if (equals(temp.name, name)) contains = true;
			temp = temp.next;
		}
		return contains;
	}
	
	// ignores case
	// params: name--the name of the person to check if they are in the graveyard
	// returns true if the person 
	// post: returns true if the graveyard contains "name"
	public boolean graveyardContains(String name) {
		AssassinNode temp = graveyard;
		boolean contains = false;
		while (temp != null) {
			if (equals(temp.name, name)) contains = true;
			temp = temp.next;
		}
		return contains;
	}
	
	// returns true if there is only one person in the kill ring
	public boolean gameOver() {
		return front.next == null;
	}
	
	// returns null if the game is not over, and returns 
	// the winner's name if the game is over
	public String winner() {
		if (!gameOver()) return null;
		return front.name;
	}
	
	// params: name--the name of the person to be killed
	// pre: the kill ring contains name, and the game is not over
	// else, throws an IllegalArgumentException() or 
	// an IllegalStateException(), respectively
	// post: adds the dead person to the graveyard, sets their killer
	// to the person who was stalking them, removes them from
	// the kill ring, and sets the killer's new target to the person the
	// dead person was stalking
	
	public void kill(String name) {
		if (!killRingContains(name)) throw new IllegalArgumentException();
		if (gameOver()) throw new IllegalStateException();
		AssassinNode temp = front;
		// Front
		if (equals(name, front.name)) {
			while (temp.next != null) temp = temp.next;
			front.killer = temp.name;
			temp = front;
			front = front.next;
			temp.next = graveyard;
			graveyard = temp;
			return;
		}
		// Middle / End
		while (!equals(temp.next.name, name)) {
			temp = temp.next;
		}
		AssassinNode temp2 = temp.next;
		temp2.killer = temp.name;
		temp.next = temp2.next;
		temp2.next = graveyard;
		graveyard = temp2;
	}
	
	// compares the two strings and ignores case
	private boolean equals(String s1, String s2) {
		return s1.toLowerCase().equals(s2.toLowerCase());
	}
}