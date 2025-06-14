`timescale 1ns / 1ps

module tb();

    reg I_clk;
    reg I_rst_n;
    reg I_key_A;
    reg I_key_B;
    reg I_key_C;
    wire [6:0] o_led;
    wire [1:0] o_dx;

    // 实例化被测模块
    sports_scorer_top  U_sports_scorer_top 
    (
        .I_clk(I_clk),
        .I_rst_n(I_rst_n),
        .I_key_A(I_key_A),
        .I_key_B(I_key_B),
        .I_key_C(I_key_C),
        .o_led(o_led),
        .o_dx(o_dx)
    );

    // 时钟信号
    initial I_clk = 0;
    always #5 I_clk = ~I_clk;  // 每 5ns 翻转一次，周期 10ns


    // 模拟测试过程
    initial begin
        // 初始值
        I_rst_n = 1'b0;
        I_key_A = 1'b0;
        I_key_B = 1'b0;
        I_key_C = 1'b0;
        
        // 复位
        #20 I_rst_n = 1'b1;
            
        //加分
        #20 I_key_B=1'b1;
        #20 I_key_B=1'b0;       
        
        #20 I_key_B=1'b1;
        #20 I_key_B=1'b0;  
        
        #20 I_key_B=1'b1;
        #20 I_key_B=1'b0;        
        //切换模式
        #20 I_key_A=1'b1;
        #20 I_key_A=1'b0; 
        
         
        //减分
        #20 I_key_B=1'b1;
        #20 I_key_B=1'b0;
        
        #20 I_key_B=1'b1;
        #20 I_key_B=1'b0; 
        
        #20 I_key_C=1'b1;
        #20 I_key_C=1'b0;              
        // 仿真结束
        #200 $finish;
    end
endmodule

    
