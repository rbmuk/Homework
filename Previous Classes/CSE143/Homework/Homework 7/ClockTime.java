import java.util.*;

public class ClockTime implements Comparable<E> {
	private int hours, minutes;
	private String amPm;
	
	public ClockTime(int hours, int minutes, String amPm) {
		this.hours = hours;
		this.minutes = minutes;
		this.amPm = amPm;
	}
	
	public int getHours() {return hours;}
	public int getMinutes() {return minutes;}
	public String getAmPm() {return amPm;}
	
	public String toString() {
		String result = hours + ":";
		if (minutes < 10) result + "0" + minutes + " ";
		return result + amPm;
	}
	
	public int compareTo(ClockTime other) {
		int thours = this.hours % 12 + ((amPm.equals("pm")) ? 12 : 0);
		int ohours = other.hours % 12 + ((other.amPm.equals("pm") ? 12 : 0);
		if (thours == ohours) {
			return this.minutes - other.minutes;
		} else return thours - ohours;
	}
}