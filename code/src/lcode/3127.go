func canMakeSquare(grid [][]byte) bool {
    for i:=0;i<len(grid)-1;i++{
        for j:=0;j<len(grid[0])-1;j++{
            flag := isSquare(i,j,grid)
            if flag{
                return true
            }
        }
    }
    return false
}

func isSquare(i,j int,grid [][]byte)bool{
    w,b := 0,0
    for row:=i;row<i + 2;row++{
        for col := j;col < j + 2;col++{
            if grid[row][col] == 'W'{
                w++
            }else{
                b++
            }
        }
    }
    if abs(w-b) == 4 || abs(w-b) == 2{
        return true
    }
    return false
}


func abs(a int)int{
    if a > 0{
        return a
    }
    return -a
}
