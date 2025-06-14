`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/06/14 11:38:49
// Design Name: 
// Module Name: score_counter
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
////////////////////////////////////////////////////////////////////////////vectored
module score_counter(
    input I_clk,
    input I_rst_n,
    input I_mode_sw,     // 按键A：切换加/减
    input I_count_key,   // 按键B：执行加/减1
    input I_clear_key,   // 按键C：清零
    output [7:0] O_bcd   // 输出BCD数，高4位十位，低4位个位
);

    reg [3:0] R_digit_l;
    reg [3:0] R_digit_h;
    reg R_mode;  // 0加分，1减分
    reg R_sw_latched;
    reg R_count_latched;

    always @(posedge I_clk or negedge I_rst_n) begin
        if (!I_rst_n) begin
            R_digit_l <= 0;
            R_digit_h <= 0;
            R_mode <= 0;
            R_sw_latched <= 0;
            R_count_latched <= 0;
        end else begin
            // 切换加/减模式
            if (I_mode_sw && !R_sw_latched) begin
                R_mode <= ~R_mode;
                R_sw_latched <= 1;
            end else if (!I_mode_sw)
                R_sw_latched <= 0;

            // 清零
            if (I_clear_key) begin
                R_digit_l <= 0;
                R_digit_h <= 0;
            end
            // 加/减分
            else if (I_count_key && !R_count_latched) begin
                if (R_mode == 0) begin
                    if (R_digit_l < 9)
                        R_digit_l <= R_digit_l + 1;
                    else if (R_digit_h < 9) begin
                        R_digit_l <= 0;
                        R_digit_h <= R_digit_h + 1;
                    end
                end else begin
                    if (R_digit_l > 0)
                        R_digit_l <= R_digit_l - 1;
                    else if (R_digit_h > 0) begin
                        R_digit_l <= 9;
                        R_digit_h <= R_digit_h - 1;
                    end
                end
                R_count_latched <= 1;
            end else if (!I_count_key)
                R_count_latched <= 0;
        end
    end

    assign O_bcd = {R_digit_h, R_digit_l};

endmodule
