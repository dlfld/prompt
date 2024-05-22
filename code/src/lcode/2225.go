func findWinners(matches [][]int) [][]int {
    loser := map[int]int{}
    gamers := map[int]int{}

    for _, matche := range matches{
        gamers[matche[0]] = 1
        gamers[matche[1]] = 1
        if _,ok := loser[matche[1]];ok{
            loser[matche[1]]++
        }else{
            loser[matche[1]] = 1
        }
    }

    res := [][]int{[]int{},[]int{}}

    for k,_ := range gamers{
        if v,ok := loser[k];ok{
            if v == 1{
                res[1] = append(res[1],k)
            }
        }else{
            res[0] = append(res[0],k)
        }
    }
    sort.Ints(res[0])
    sort.Ints(res[1])
    return res
}
