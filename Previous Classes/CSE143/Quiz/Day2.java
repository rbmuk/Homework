import java.util.*;

public class Day2 {
   public static void main(String[] args) {
      List<String> l = new ArrayList<>();
      l.add("laughing");
      l.add("out");
      l.add("loud");
      System.out.println(acronymFor(l));
      
      Set<Integer> set = new HashSet<>();
      set.add(1);
      set.add(2);
      set.add(3);
      set.add(4);
      set.add(5);
      System.out.println(set);
      Set<Integer> evens = removeEven(set);
      System.out.println(set);
      System.out.println(evens);
   }
   
   public static String acronymFor(List<String> l) {
      if (l.isEmpty()) return "";
      String acr = "";
      for (String s : l) {
         acr += Character.toUpperCase(s.charAt(0));
      }
      return acr;
   }
   
   public static Set<Integer> removeEven(Set<Integer> s) {
      Set<Integer> evens = new TreeSet<>();
      /*Iterator<Integer> itr = s.iterator();
      while (itr.hasNext()) {
         Integer i = itr.next();
         if (i % 2 == 0) {
            evens.add(i);
            itr.remove();
         }
      }*/
      for (var itr = s.iterator(); itr.hasNext();) {
         Integer i = itr.next();
         if (i % 2 == 1) continue;
         evens.add(i);
         itr.remove();
      }
      return evens;
   }
   
   public static Stack<Integer> queueToStack(Queue<Integer> q, Stack<Integer> s) {
   		while (!q.isEmpty()) {
			int n = q.remove();
			s.push(n);
		}
	}
}