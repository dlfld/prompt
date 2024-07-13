class Solution {
    private static int res = 0;
    public int crackNumber(int ciphertext) {
        res = 0;
        dfs(0,String.valueOf(ciphertext));
        return res;
    }

    public static void dfs(int start,String ciphertext){
        if(start > ciphertext.length()){
            res++;
            return;
        }
        dfs(start+1,ciphertext);
//        表示不超过字母上限
        if(start+1 < ciphertext.length()){
            int item = Integer.parseInt(ciphertext.substring(start,start+2));
            if (item < 26 && item > 9){
                dfs(start + 2,ciphertext);
            }
          
        }
    }
}
