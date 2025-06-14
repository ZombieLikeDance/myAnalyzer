`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/06/14 11:38:04
// Design Name: 
// Module Name: light_show
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module light_show(
    input        I_clk,
    input        I_rst_n,
    input  [7:0] I_show_num,  // 高4位十位，低4位个位（BCD）
    output [6:0] o_led,       // 段选，共阳极，高电平点亮
    output [1:0] o_dx         // 位选，低电平有效
);

    parameter C_COUNTER_NUM = 100_000;
    reg [3:0] R_temp;
    reg [1:0] R_dx_temp;
    reg [31:0] R_counter;

    always @(posedge I_clk or negedge I_rst_n) begin
        if (!I_rst_n) begin
            R_dx_temp <= 2'b01;
            R_temp <= I_show_num[3:0];
            R_counter <= 0;
        end else begin
            if (R_counter >= C_COUNTER_NUM) begin
                R_counter <= 0;
                if (R_dx_temp == 2'b01) begin
                    R_dx_temp <= 2'b10;
                    R_temp <= I_show_num[7:4];
                end else begin
                    R_dx_temp <= 2'b01;
                    R_temp <= I_show_num[3:0];
                end
            end else begin
                R_counter <= R_counter + 1;
            end
        end
    end

    // 共阳极段码：高电平点亮
    reg [6:0] seg;
    always @(*) begin
        case (R_temp)
            4'd0: seg = 7'b1111110;
            4'd1: seg = 7'b0110000;
            4'd2: seg = 7'b1101101;
            4'd3: seg = 7'b1111001;
            4'd4: seg = 7'b0110011;
            4'd5: seg = 7'b1011011;
            4'd6: seg = 7'b1011111;
            4'd7: seg = 7'b1110000;
            4'd8: seg = 7'b1111111;
            4'd9: seg = 7'b1111011;
            default: seg = 7'b0000000;
        endcase
    end

    assign o_led = seg;
    assign o_dx = R_dx_temp;

endmodule
