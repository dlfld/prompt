func numberOfWeeks(milestones []int) int64 {
    maxVal,sum := getMax(milestones)
    if maxVal > sum - maxVal + 1{
        return 2 * (sum - maxVal) + 1
    }else{
        return sum
    }
}

func getMax(nums []int)(int64,int64){
    res := int64(nums[0])
    var sum int64 = 0
    for _,v:=range nums{
        if res < int64(v){
            res = int64(v)
        }
        sum += int64(v)
    }    
    return res,sum
}


