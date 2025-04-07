import java.util.Arrays;

class Solution {
    public boolean canPartition(int[] nums) {
        Arrays.sort(nums);
        int sum = Arrays.stream(nums).sum();
        if(sum % 2 != 0){
            return false;
        }
        int mid = sum / 2;
        int [][]dp = new int[nums.length][mid + 1];
        dp[0][0] = nums[0];
        for (int i = 0; i < nums.length; i++) {
            for (int j = 1; j < mid + 1; j++) {
                if(i == 0){
                    dp[i][j] = nums[0];
                    continue;
                }
                if(j < nums[i]){
                    dp[i][j] = dp[i-1][j];
                }else{
                    dp[i][j] = Math.max(dp[i-1][j],dp[i-1][j - nums[i]] + nums[i]);
                }
                if(dp[i][j] == mid){
                    return true;
                }
            }
        }
        return false;
    }
}
