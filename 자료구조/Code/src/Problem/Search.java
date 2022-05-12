package Problem;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;

public class Search {
    private class Pos{
        int row;
        int col;
        Pos(int row, int col){
            this.row = row;
            this.col = col;
        }
    }
    public Search(){
    }

    int[] dcol = {0,1,0,-1};
    int[] drow = {-1,0,1,0};

//    public int[] BFS_solution(int[][] picture){
//
//    }

    public int[] BFS_solution(int m, int n, int[][] pictrue){
        int number_of_area = 0;
        int max_size_of_area = 0;
        int[] result = {Integer.MAX_VALUE, Integer.MIN_VALUE};
        boolean[][] visited = new boolean[m][n];
        Queue<Pos> q = new LinkedList<>();

        int area_number = 0;
        for(int i = 0; i< pictrue.length; i++){
            for(int j = 0; j < pictrue[0].length; j++){
                if(pictrue[i][j] == 0)  continue;
                if(!visited[i][j]) {
                    q.add(new Pos(i, j));
                    number_of_area++;
                    max_size_of_area = 0;
                    area_number = pictrue[i][j];
                    visited[i][j] = true;
                }
                while(!q.isEmpty()){
                    int cx = q.peek().col;
                    int cy = q.peek().row;
                    q.poll();
                    max_size_of_area++;
                    for(int d = 0; d < 4; d++){
                        int ny = cy + drow[d];
                        int nx = cx + dcol[d];
                        if(nx < 0 || nx >= n || ny < 0 || ny >= m)  continue;
                        if(visited[ny][nx]) continue;
                        if(pictrue[ny][nx] != area_number)   continue;
                        visited[ny][nx] = true;
                        q.add(new Pos(ny,nx));
                    }
                }
                result[1] = Math.max(result[1], max_size_of_area);
            }
        }
        result[0] = number_of_area;
        return result;
    }

    public int[] DFS_solution(int[][] map){
        int[] res = {Integer.MAX_VALUE, Integer.MIN_VALUE};
        boolean[][] visited = new boolean[map.length][map[0].length];
        int n_group = 0;
        for(int i = 0; i < map.length; i++){
            for(int j = 0; j < map[0].length; j++){
                if(!visited[i][j] && map[i][j] > 0){
                    int group_size = 0;
                    n_group++;
                    group_size = DFS(i,j, map, visited, map[i][j]);
                    res[1] = Math.max(res[1], group_size);
                }
            }
        }
        res[0] = n_group;
        return res;
    }
    private int DFS(int row, int col, int[][] map, boolean[][] visited, int class_n){
        if(visited[row][col])   return 0;
        visited[row][col] = true;
        int g_size = 1;
        for(int i= 0; i < 4; i++){
            int nrow = row + drow[i];
            int ncol = col + dcol[i];
            if(ncol < 0 || ncol >= map[0].length || nrow < 0 || nrow >= map.length)  continue;
            if(visited[nrow][ncol]) continue;
            if(class_n == map[nrow][ncol]){
                g_size += DFS(nrow,ncol, map, visited ,class_n);
            }
        }
        return g_size;
    }


    public int[] DFS_solution_v(int[][] map){
        int[][] visited = new int[map.length][map[0].length];
        int n_group = 0;

        int[] result = {0, 0};
        for(int i = 0; i < map.length; i++){
            for(int j = 0; j < map[0].length; j++){
                if(map[i][j] == 1 && visited[i][j] == 0){
                    result[1] = Math.max(DFS_vector(i,j, map, visited, ++n_group), result[1]);
                }
            }
        }
        result[0] = n_group;

        for(int[] l : visited){
            System.out.println(Arrays.toString(l));
        }
        return result;
    }
    public int DFS_vector(int start_row, int start_col, int[][] map, int[][] visited, int n_group){
        Stack<Pos> st = new Stack<>();
        st.push(new Pos(start_row, start_col));
        int size = 0;
//        System.out.println(start_row + ", " + start_col);

        while(!st.empty()){
            Pos cp = st.pop();
            int row = cp.row;
            int col = cp.col;
            if(visited[row][col] == 0){
                size++;
            }
            visited[row][col] = n_group;

            for(int i = 0; i < 4; i++){
                int n_row = row + drow[i];
                int n_col = col + dcol[i];

                if(n_row < 0 || n_row >= map.length || n_col < 0 || n_col >= map[0].length) continue;
                if(visited[n_row][n_col] == 0 && map[n_row][n_col] == 1)
                    st.push(new Pos(n_row, n_col));
            }
        }
        System.out.println(n_group + ", " +size);
        return size;
    }
}
