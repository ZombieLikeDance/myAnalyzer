`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/06/14 11:39:39
// Design Name: 
// Module Name: sports_scorer_top
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module sports_scorer_top(
    input I_clk,
    input I_rst_n,
    input I_key_A,  // 模式切换
    input I_key_B,  // 计数
    input I_key_C,  // 清零
    output [6:0] o_led,
    output [1:0] o_dx
);

    wire W_mode_sw, W_count_key, W_clear_key;
    wire [7:0] W_bcd;

    // 三个按键消抖
    key_rebounce U_A (.I_clk(I_clk), .I_rst_n(I_rst_n), .I_key_in(I_key_A), .o_key_out(W_mode_sw));
    key_rebounce U_B (.I_clk(I_clk), .I_rst_n(I_rst_n), .I_key_in(I_key_B), .o_key_out(W_count_key));
    key_rebounce U_C (.I_clk(I_clk), .I_rst_n(I_rst_n), .I_key_in(I_key_C), .o_key_out(W_clear_key));

    // 分数计数器
    score_counter U_counter (
        .I_clk(I_clk),
        .I_rst_n(I_rst_n),
        .I_mode_sw(W_mode_sw),
        .I_count_key(W_count_key),
        .I_clear_key(W_clear_key),
        .O_bcd(W_bcd)
    );

    // 数码管显示
    light_show U_show (
        .I_clk(I_clk),
        .I_rst_n(I_rst_n),
        .I_show_num(W_bcd),
        .o_led(o_led),
        .o_dx(o_dx)
    );

endmodule

