import java.util.*;

public class DayUnknown2 {
	public static void main(String[] args) {
		List<Integer> l = Arrays.asList(5, 4, 3, 2, 1, 0);
		reverse3(l);
		System.out.println(l);
		Map<String, List<Integer>> m = new HashMap<>();
		m.put("cse143", Arrays.asList(42, 17, 42, 42));
		m.put("goodbye", Arrays.asList(3, 10, -5));
		m.put("hello", Arrays.asList(16, 8, 0, 0, 106));
		Map<String, List<Integer>> copy = deepCopy(m);
		m.remove("cse143");
		System.out.println(m);
		System.out.println(copy);
	}

	public static void reverse3(List<Integer> list) {
		for (int i = 0; i + 2 < list.size(); i += 3) {
			int temp = list.get(i);
			list.set(i, list.get(i+2));
			list.set(i+2, temp);
		}
	}
	
	public boolean containsAll(Set<Integer> s1, Set<Integer> s2) {
		for (int i : s2) {
			if (!s1.contains(i)) return false;
		}
		return true;
	}
	
	public static Map<String, List<Integer>> deepCopy(Map<String, List<Integer>> map) {
		Map<String, List<Integer>> copy = new TreeMap<>();
		for (String key : map.keySet()) {
			List<Integer> t = new ArrayList<>();
			for (int i : map.get(key)) {
				t.add(i);
			}
			copy.put(key, t);
		}
		return copy;
	}
}