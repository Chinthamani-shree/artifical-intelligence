public class AlphaBetaPruning { 
  
    // Initial values of Aplha and Beta 
    static int MAX = 1000; 
    static int MIN = -1000; 
   
    // Returns optimal value for 
    // current player (Initially called 
    // for root and maximizer) 
    static int minimax(int depth, int nodeIndex,  
        Boolean maximizingPlayer, 
        int values[], int alpha, 
        int beta) { 
        // Terminating condition. i.e  
        // leaf node is reached 
        if (depth == 4) 
            return values[nodeIndex]; 
   
        if (maximizingPlayer) { 
            int best = MIN; 
   
            // Recur for left and right children 
            for (int i = 0; i < 2; i++) { 
                int val = minimax(depth + 1, nodeIndex * 2 + i, 
                    false, values, alpha, beta); 
                best = Math.max(best, val); 
                alpha = Math.max(alpha, best); 
   
                // Alpha Beta Pruning 
                if (beta <= alpha) 
                    break; 
             } 
             return best; 
        } else { 
            int best = MAX; 
            // Recur for left and 
            // right children 
            for (int i = 0; i < 2; i++) {  
                int val = minimax(depth + 1, nodeIndex * 2 + i, 
                    true, values, alpha, beta); 
                best = Math.min(best, val); 
                beta = Math.min(beta, best); 
   
                // Alpha Beta Pruning 
                if (beta <= alpha) 
                    break; 
            } 
            return best; 
        } 
    } 
 
    public static void main (String[] args) { 
        int values[] = {3, 17, 2, 12, 15, 25, 0, 2, 5, 3, 2, 14}; 
        System.out.println("The optimal value : " + 
                            minimax(0, 0, true, values, MIN, MAX)); 
    } 
}
