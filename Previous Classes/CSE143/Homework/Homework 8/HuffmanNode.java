import java.util.*;

public class HuffmanNode implements Comparable<HuffmanNode> {
	
	public final int c, freq;
	public HuffmanNode left, right;
	
	public HuffmanNode(int c, int freq) {
		this.c = c;
		this.freq = freq;
	}
	
	public HuffmanNode(int freq, HuffmanNode left, HuffmanNode right) {
		this(0, freq);
		this.left = left;
		this.right = right;
	}
	
	public boolean isLeaf() {
		return c != 0;
	}
	
	public int compareTo(HuffmanNode other) {
		return freq-other.freq;
	}
}