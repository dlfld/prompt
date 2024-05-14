func minimumRounds(tasks []int) int {
    dict := map[int]int{}
    for _,v := range tasks{
        value,ok := dict[v]
        if ok {
            dict[v] = value + 1
        }else{
            dict[v] = 1
        }
    }
    res := 0
    for _,v := range dict{
        if v <= 1{
            return -1
        }
        if v % 3 == 0{
            res += v / 3
        }else if v % 3 == 1{
            res += (v - 4)/3 + 2
        }else{
            res += v / 3 + 1
        }
    }
    return res
}
