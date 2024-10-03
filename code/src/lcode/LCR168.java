import java.util.HashSet;
import java.util.PriorityQueue;

class Solution {
    public int nthUglyNumber(int n) {
        int[] nums = {2, 3, 5};
        PriorityQueue<Long> heap = new PriorityQueue<>();
        HashSet<Long> set = new HashSet<>();
        heap.add(1L);
        int res = 0;
        while (n-- > 0) {
            long cur = heap.poll();
            res = (int) cur;
            for (int num : nums) {
                long item = cur * num;
                if(!set.contains(item)){
                    heap.add(item);
                    set.add(item);
                }
            }
        }
        return res;
    }
}
