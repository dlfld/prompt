class Solution {
    public int[] sockCollocation(int[] sockets) {
        int item = 0;
        for (int socket : sockets) {
            item = item ^ socket;
        }
        int[] res = new int[2];
        // 找到最低位的1
        int lowBit = item & (-item);
        int res1 = 0, res2 = 0;
        for (int socket : sockets) {
            if((socket & lowBit) == 0){
                res1 ^= socket; 
            }else{
                res2 ^= socket;
            }
        }
        res[0] = res1;
        res[1] = res2;
        return res;
    }
}
