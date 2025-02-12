public class GuitarString {

	Queue<Double> ringBuffer;
	public GuitarString(double frequency) {
		if (frequency <= 0) throw new IllegalArgumentException();
		int N = (int) (StdAudio.SAMPLE_RATE / frequency);
		if (N < 2) throw new IllegalArgumentException();
		ringBuffer = new LinkedList<>();
		for (int i = 0; i < N; ++i) {
			ringBuffer.add(0);
		}
	}
	
	public GuitarString(double[] init) {
		if (init.length < 2) throw new IllegalArgumentException();
		ringBuffer = new LinkedList<>();
		for (int i = 0; i < init.length; ++i) {
			ringBuffer.add(init[i]);
		}
	}
}