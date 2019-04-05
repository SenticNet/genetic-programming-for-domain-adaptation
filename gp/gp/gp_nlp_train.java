import java.util.Random;
import java.io.*;
import java.lang.*;
import java.util.*;
import org.jgap.gp.*;


 public class gp_nlp_train{
 public static int indx1 = 9;  

 public static void main(String[] args) {

	                                   Random rng = new Random(123);
                                           int neu = 15;
	                                   double[][] layer_input; 
									   int[] testindx = new int[neu];
									   indx1 = Integer.parseInt(args[0]);
	                                   int[] label;

                                       String dirgp = "../neu_gp"+indx1;

                       									   // read in testindx
									   try{
	                                     FileReader fr = new FileReader(dirgp+"/test_ind"+indx1);
	                                     BufferedReader br = new BufferedReader(fr);

	                                     String s = "";
	                                     s=br.readLine();
										 int i=0;
										 do
	                                     {
	                                         testindx[i]=Integer.parseInt(s);
											 i++;
									     }
										  while((s=br.readLine())!=null);
									     fr.close();
										 br.close();
                                       }catch(Exception e){}

									   try{
	                                   Runtime r = Runtime.getRuntime();
                                       Process p = r.exec("del "+dirgp+"/log"+".txt");
                                       p.waitFor();
	                                   p = r.exec("del "+dirgp+"/log_test"+".txt");
                                       p.waitFor(); 
                                           p = r.exec("del "+dirgp+"/solution"+".txt");
                                       p.waitFor(); 

									   }catch(Exception e){}

	                                   int n_ins = neu; // number of dimensions
	                                   double[] allerr = new double[neu+1];

                                       for(int indx=1; indx<2; indx++){

									   if(testindx[indx-1]>0){


	                                   try{
									     
	                                     FileReader fr = new FileReader(dirgp+"/nlp_neu"+indx+".txt");
	                                     BufferedReader br = new BufferedReader(fr);
	                                     String s = "";
	                                     int j =0;
	                                     s=br.readLine();                                    
	                                     String[] cell = s.split(",");
	                                     
										// System.out.println(s);

	                                     layer_input = new double[cell.length][n_ins];
	                                     label = new int[cell.length];
	                                     
										 try{
	                                     do
	                                     {	    
										     //System.out.println(s);                  	 
	                                         cell = s.split(",");
	                                         for(int i=0; i<cell.length; i++){
             	                   				if(j<n_ins)
	                                               layer_input[i][j]=Double.parseDouble(cell[i]);
	                   						    if(j==n_ins)	
	                   						       label[i]=Integer.parseInt(cell[i]);
	                                            }
	                                         j++;
	                                     }
	                                     while((s=br.readLine())!=null);
	                                    
										 }catch(Exception e){}

	                                    int npop = 1000, ngen = 2000;
	          
                                        FileWriter ostreamg = new FileWriter("dnn.conf");
                                        ostreamg.write("presentation: DNN\n");
                                        ostreamg.write("return_type: DoubleClass\n");
                                        ostreamg.write("num_input_variables: "+layer_input[0].length+"\n");
                                        String var_name = "variable_names:";
                                        for(int i1=1; i1<=layer_input[0].length+1; i1++)var_name+=" A"+i1;
                                        ostreamg.write(var_name+"\n");
                                        ostreamg.write("output_variable: "+layer_input[0].length+"\n");
                                        ostreamg.write("functions: Multiply,Divide,Add,Subtract,Log,Sqrt,XorD,Step,Square,Sign,Sigmoid,RoundD,OrD,NotD,ModuloD,LoopD,Logistic,LesserThanOrEqualD,LesserThanD,IfLessThanOrEqualD,Gamma,Gaussian,Cube\n");
                                        ostreamg.write("terminal_range: -500 500\n");
                                      //  ostreamg.write("terminal_wholenumbers: true\n");                                       
										ostreamg.write("max_init_depth: "+5+"\n");
                                        ostreamg.write("population_size: "+npop+"\n");					 
                                        ostreamg.write("max_crossover_depth: "+5+"\n");
                                        ostreamg.write("num_evolutions: "+ngen+"\n");
										ostreamg.write("show_progression: false\n"); 										
                                        ostreamg.write("max_nodes: "+20+" \n");
                                        ostreamg.write("mutation_prob: 0.9\n");
                                        ostreamg.write("crossover_prob: 0.1\n");
                                        ostreamg.write("result_precision: 5\n");
                                       // ostreamg.write("hits_criteria: 0.5\n");
                                        ostreamg.write("validation_pct: .05\n");
                                        
                                        ostreamg.write("stop_criteria_fitness: 0.001\n");
                                        ostreamg.write("error_method: meanError\n");
                                        ostreamg.write("adf_arity: 0\n");
                                        ostreamg.write("program_creation_max_tries: 20\n");
                                        ostreamg.write("show_progression: False\n");  
                                        ostreamg.write("show_similar: False\n");
                                        ostreamg.write("show_all_generations: False\n");  
                                        ostreamg.write("show_results: False\n");  
                                        ostreamg.write("adf_type: double\n");
                                        ostreamg.write("data\n");
                                        ostreamg.close();
         
                                        ostreamg = new FileWriter("dnn.conf",true);
									
                                        int starti = (int)(layer_input.length-testindx[indx-1]);   
										

                                        for(int i=0; i<layer_input.length; i++){
                                              
                                              String line = "";
      										  for(j=0; j<layer_input[0].length; j++)line+=layer_input[i][j]+",";
                                              if(i<starti)line+=label[i]+"\n";
											  else
											    line+="?\n";
                                              ostreamg.write(line);
                                        }
                                        ostreamg.close();
                        
                                         ostreamg = new FileWriter(dirgp+"/log.txt",true);
                                                      ostreamg.write("starting neuron:"+indx +"\n");
                                                      ostreamg.close();  

                                         ostreamg = new FileWriter(dirgp+"/log_test.txt",true);
                                                      ostreamg.write("starting neuron:"+indx +"\n");
                                                      ostreamg.close();

                                         ostreamg = new FileWriter(dirgp+"/solution.txt",true);
                                                      ostreamg.write("starting neuron:"+indx +"\n");
                                                      ostreamg.close();   

          
                                                Thread thread = new Thread(){
                                                  public void run(){
                                                       try{
                                                       SymbolicRegression.indx1 = indx1;
													   SymbolicRegression.dirgp = "../neu_gp"+indx1;	
                                                       SymbolicRegression.main(new String[]{"dnn.conf"});
             
                                                       }catch(Exception e2){}                                             
                                                    }
                                                 };
                                                 thread.start();
                                                 thread.join();    
                                              
							
                                           
                     }
                     catch(Exception e){}
                    } // end of err if 
                  } // end of indx for

				
           }
}									
