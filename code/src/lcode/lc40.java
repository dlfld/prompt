import java.util.Arrays;

class Solution {
    public int maxmiumScore(int[] cards, int cnt) {
        Arrays.sort(cards);
        int res = 0;
        int minodd = -1;
        int mineven = -1;
        for(int i=cards.length-1;i>=cards.length - cnt;i--){
            res += cards[i];
            if(cards[i] % 2 == 0){
                minodd = cards[i];
            }else{
                mineven = cards[i];
            }
        }
        if (res % 2 == 0){
            return res;
        }
//        res = 0;
        int ans = 0;
        for(int i = cards.length - cnt - 1;i>=0;i--){
            if (cards[i] % 2 == 0){
                if(mineven != -1){
         ans = Math.max(ans,res - mineven + cards[i]);
                break;
                }
       
            }
        }
        for(int i = cards.length - cnt - 1;i>=0;i--){
            if (cards[i] % 2 != 0){
                if(minodd != -1){
                ans = Math.max(ans,res - minodd + cards[i]);
                break;
                }

            }
        }
        
        return ans;
    }
}
