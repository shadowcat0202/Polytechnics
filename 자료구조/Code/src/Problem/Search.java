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

    int[] dcol = { 0,1,0,-1};
    int[] drow = {-1,0,1,0};

//    public int[] BFS_solution(int[][] picture){
//
//    }


    public int[] solution_use_recursive(int[][] map){
        int[][] visited = new int[map.length][map[0].length];
        int n_group = 0;
        int max_area = 0;
        for(int i = 0; i < map.length; i++){
            for(int j = 0; j < map[0].length; j++){
                if(map[i][j] != 0 && visited[i][j] == 0) {
                    max_area = Math.max(DFS_recursive(i, j, map, visited, map[i][j], ++n_group), max_area);
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int[] ints : visited) {
            for (int anInt : ints) {
                sb.append(anInt).append(" ");
            }
            sb.append("\n");
        }
        System.out.println(sb.toString());
        return new int[] {n_group, max_area};
    }

    // 큐 혹은 스텍을 사용해서 해결
    public int[] solution_use_Queue_Stack(int[][] map){
        int[][] visited = new int[map.length][map[0].length];
        int n_group = 0;
        int max_area = 0;

        for(int row = 0; row < map.length; row++){
            for(int col = 0; col < map[0].length; col++){
                if(map[row][col] != 0 && visited[row][col] == 0) {
                    max_area = Math.max(vector(row, col, map, visited, ++n_group), max_area);   // 스텍을 이용한 방법
//                    max_area = Math.max(queue(i, j, map, visited, ++n_group), max_area);    // 큐를 이용한 방법
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int[] ints : visited) {
            for (int anInt : ints) {
                sb.append(anInt).append(" ");
            }
            sb.append("\n");
        }
        System.out.println(sb.toString());
        return new int[] {n_group, max_area};
    }

    private int vector(int startRow, int startCol, int[][] input, int[][] visited, int n_group){
        int area_size = 0;
        Stack<Pos> stack = new Stack<>();
        Pos start = new Pos(startRow, startCol);
        stack.push(start);

        while(!stack.isEmpty()){
            Pos nowPos = stack.pop();
            if(visited[nowPos.row][nowPos.col] == 0)    area_size++;
            visited[nowPos.row][nowPos.col] = n_group;


            for(int i = 0; i < 4; i++){
                int n_row = nowPos.row + drow[i];
                int n_col = nowPos.col + dcol[i];
                if(n_row < 0 || n_row >= visited.length || n_col < 0 || n_col >= visited.length)    continue;
                if(visited[n_row][n_col] > 0)   continue;
                if(input[n_row][n_col] == 0)    continue;

                stack.push(new Pos(n_row, n_col));
            }
        }
        return area_size;
    }
    private int queue(int startRow, int startCol, int[][] input, int[][] visited, int n_group){
        int area_size = 0;
        Queue<Pos> que = new LinkedList<>();
        Pos start = new Pos(startRow, startCol);
        que.offer(start);

        while(!que.isEmpty()){
            Pos nowPos = que.poll();
            if(visited[nowPos.row][nowPos.col] == 0)    area_size++;
            visited[nowPos.row][nowPos.col] = n_group;


            for(int i = 0; i < 4; i++){
                int n_row = nowPos.row + drow[i];
                int n_col = nowPos.col + dcol[i];
                if(n_row < 0 || n_row >= visited.length || n_col < 0 || n_col >= visited.length)    continue;
                if(visited[n_row][n_col] > 0)   continue;
                if(input[n_row][n_col] == 0)    continue;

                que.offer(new Pos(n_row, n_col));
            }
        }
        return area_size;
    }

    // 재귀
    private int DFS_recursive(int row, int col, int[][] map, int[][] visited, int prev, int n_group){
        int tempCnt = 1;
        if(map[row][col] != prev)   return 0;

        visited[row][col] = n_group;

        for(int i = 0; i < 4; i++){
            int n_row = row + drow[i];
            int n_col = col + dcol[i];

            if(n_row < 0 || n_row >= visited.length || n_col < 0 || n_col >= visited.length)    continue;
            if(visited[n_row][n_col] > 0)   continue;
            if(map[n_row][n_col] != prev)    continue;

            tempCnt += DFS_recursive(n_row, n_col, map, visited, prev, n_group);
        }
        return tempCnt;
    }


}
