- Website
   - Get manual segment on website
- Figure out how to transfer learn in SAM
   - High prio how to label images for transfer learning
- Dot counting through SAM
- Z stack in SAM
- Linking nucleus and cytoplasm together into one cell
	- Use "center of mass"
		- if nucleus within bounding box of cytoplasm they are linked
		- need to know which segment is nucleus and which is cyto


- Automatic ND2 Parsing
    - ?
    - ~~Must be some metadata I Can exploit~~
- Memory Optim
    - Save ND2 info into folder and retrieve when needed
- Dot Counting
    - CPU Testing v GPU Time
    - ~~Quilt process generalizable~~
    - ~~Try filling background with noise for performance~~
    - ~~Adjust pred iou thresh for accuracy~~
    - ~~Concede accuracy for speed~~
- Manual Seg
    - Cell ID on Image
    - Overlay of previous nuc seg for cyto
- Test run z smashed
    - Test z smash counting
    - Test z smash manual segmentations



























- Progress Bar
    - Ask Andy
