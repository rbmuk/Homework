public abstract class AbstractIntList implements IntList {
	public void addAll(IntList other) {
	/*
		for (int i = 0; i < other.size(); ++i) {
			add(other.get(i));
		}
	
		Iterator<Integer> it = other.iterator();
		while (it.hasNext())
			add(it.next());
	*/
		for (int n : other)
			add(n);
	}
	
	public void removeAll(IntList other) {
		Iterator<Integer> it = iterator();
		while (it.hasNext())
			if (other.contains(it.next())
				it.remove();
	}
}