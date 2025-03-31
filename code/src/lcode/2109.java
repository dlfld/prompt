class Solution {
    public String addSpaces(String s, int[] spaces) {
        StringBuilder res = new StringBuilder();
        Arrays.sort(spaces);
        int cur = 0;

        for (int i = 0; i < s.length(); i++) {
            if (cur < spaces.length && i == spaces[cur]) {
                res.append(' ');
                cur++;
            }
            res.append(s.charAt(i));
        }
        return res.toString();
    }
}
