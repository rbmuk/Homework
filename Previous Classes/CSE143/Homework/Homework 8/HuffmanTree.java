import java.util.*;
import java.io.*;

public class HuffmanTree {
	
	private HuffmanNode overallRoot;
	
	public HuffmanTree(int[] count) {
		Queue<HuffmanNode> pq = new PriorityQueue<>();
		for (int i = 0; i < count.length; ++i) 
			if (count[i] > 0)
			{
				System.out.println(i + " " + count[i]);
				pq.add(new HuffmanNode(i, count[i]));		
			}
		pq.add(new HuffmanNode(256, 1));
		
		while (pq.size() > 1) {
			HuffmanNode first = pq.remove();
			HuffmanNode second = pq.remove();
			HuffmanNode n = new HuffmanNode(first.freq + second.freq, first, second);
			pq.add(n);
		}
		overallRoot = pq.remove();
	}
	
	public HuffmanTree(Scanner input) {
		while (input.hasNext()) {
			int c = Integer.parseInt(input.nextLine());
			String path = input.nextLine();
			overallRoot = construct(c, path, 0, overallRoot);
		}
	}
	
	private HuffmanNode construct(int c, String path, int at, HuffmanNode root) {
		if (root == null) {
			root = new HuffmanNode(0, null, null);
		}
		if (path.charAt(at) == 0) {
			root.left = construct(c, path, at+1, root.left);
		} else {
			
		}
		
		if (at == path.length() - 1) {
			if (path.charAt(at) == 0) {
				root.left = new HuffmanNode(c, 0);
			else root.right = new HuffmanNode(c, 0);
		}
	}
	
	public void decode(BitInputStream input, PrintStream output, int eof) {
		decode(input, output, eof, overallRoot);
	}
	
	private void decode(BitInputStream input, PrintStream output, int eof, HuffmanNode root) {
		if (!root.isLeaf()) {
			int bit = input.readBit();
			if (bit == 0) decode(input, output, eof, root.left);
			else decode(input, output, eof, root.right);
		} else {
			if (root.c != eof) {
				output.print((char)root.c);
				decode(input, output, eof, overallRoot);
			}
		}
		return false;
	}
	
	public void write(PrintStream output) {
		write(output, overallRoot, "");
	}
	
	private void write(PrintStream output, HuffmanNode root, String path) {
		if (root.isLeaf()) {
			output.println(root.c);
			output.println(path);
		} else {
			path += "0";
			write(output, root.left, path);
			path = path.substring(0, path.length() - 1);
			path += "1";
			write(output, root.right, path);
		}
	}
}