import java.util.Arrays;

class Solution {
    public double[] statisticsProbability(int num) {
        double item = 1.0 / 6;
        double[][] dp = new double[num + 1][6 * num + 1];
        // 当num为1的时候，初始化每一个值的可能性
        for (int i = 1; i < 7; i++) {
            dp[1][i] = item;
        }
        for (int i = 1; i <= num; i++) {
            // 遍历骰子个数
            for (int j = i; j <= i * 6; j++) {
                // 遍历当前个数的骰子能走出来的值
                // dp[i][j] 的含义： 当，当前轮次的骰子值为j时，概率为多少
                for (int k = 1; k < 7 && j - k > 0; k++) {
                    // 遍历当前骰子从1 - 6
                    dp[i][j] += dp[i - 1][j - k] * item;
                }
            }
        }
        return Arrays.copyOfRange(dp[num], num * 1, num * 6 + 1);
    }
}
